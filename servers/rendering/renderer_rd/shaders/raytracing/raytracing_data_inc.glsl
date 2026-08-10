// Shared data structures for raytracing shaders.
// Requires: GL_EXT_buffer_reference, GL_ARB_gpu_shader_int64, raytracing_inc.glsl (for OFFSET_NONE/FLAG_*)

// ============================================================================
// BUFFER REFERENCES
// ============================================================================
layout(buffer_reference, std430) readonly buffer FloatBuffer {
	float v[];
};
layout(buffer_reference, std430) readonly buffer Uint32Buffer {
	uint v[];
};

// ============================================================================
// GEOMETRY DATA (matches C++ RT_GeometryData, 128 bytes)
// ============================================================================
struct GeometryData {
	uint64_t vertex_address;
	uint64_t attribute_address;
	uint64_t index_address;

	uint vertex_count;
	uint position_stride;
	uint normal_byte_offset;
	uint normal_stride;
	uint tangent_byte_offset;
	uint tangent_stride;

	uint attribute_stride;
	uint uv_byte_offset;
	uint uv_scale_packed; // fp16 x, fp16 y
	uint index_format;
	uint primitive_count;
	uint flags;
	float aabb_size_x;
	float aabb_size_y;
	float aabb_size_z;

	// Byte offset of the vertex color attribute inside `attribute_stride`,
	// or OFFSET_NONE if the mesh has no vertex colors.
	uint color_byte_offset;

	float aabb_pos_x;
	float aabb_pos_y;
	float aabb_pos_z;

	// For deformed geometry: previous-frame position buffer used for motion vectors.
	uint prev_vertex_address_lo;
	uint prev_vertex_address_hi;

	uint _pad[5];
};

void get_aabb_compression_xforms(GeometryData geom, out mat4 aabb_xform, out mat4 inv_aabb_xform) {
	if ((geom.flags & FLAG_COMPRESSED) == 0u) {
		aabb_xform = mat4(1.0);
		inv_aabb_xform = mat4(1.0);
		return;
	}
	vec3 aabb_sz = vec3(geom.aabb_size_x, geom.aabb_size_y, geom.aabb_size_z);
	vec3 aabb_po = vec3(geom.aabb_pos_x, geom.aabb_pos_y, geom.aabb_pos_z);
	aabb_xform = mat4(
			vec4(aabb_sz.x, 0.0, 0.0, 0.0),
			vec4(0.0, aabb_sz.y, 0.0, 0.0),
			vec4(0.0, 0.0, aabb_sz.z, 0.0),
			vec4(aabb_po, 1.0));
	vec3 inv_sz = 1.0 / aabb_sz;
	inv_aabb_xform = mat4(
			vec4(inv_sz.x, 0.0, 0.0, 0.0),
			vec4(0.0, inv_sz.y, 0.0, 0.0),
			vec4(0.0, 0.0, inv_sz.z, 0.0),
			vec4(-aabb_po * inv_sz, 1.0));
}

// ============================================================================
// PER-INSTANCE MOTION DATA (matches C++ RT_InstanceMotionData, 48 bytes)
// ============================================================================
struct InstanceMotionData {
	float prev_xform[12]; // Previous object-to-world (mat3x4, transposed 3x4)
};

// ============================================================================
// MATERIAL DATA (matches C++ layout, 128 bytes)
// ============================================================================
struct MaterialData {
	uint albedo_texture_idx;
	uint normal_texture_idx;
	uint orm_texture_idx;
	uint emission_texture_idx;

	vec4 albedo_color;
	vec3 emission_color;
	float emission_strength;

	float metallic;
	float roughness;
	float ao_strength;
	uint flags; // Bit 0: has_normal_map, Bit 1: has_emission

	vec2 uv1_scale; // UV1 scale (default 1,1)
	vec2 uv1_offset; // UV1 offset (default 0,0)

	float normal_map_depth; // Normal map strength (default 1.0)
	float specular; // Dielectric specular [0..1], default 0.5 -> F0 = 0.04.
	uint64_t uniform_address; // BDA for custom shader uniform buffer (0 = none)

	// Pathtracer material contract (see render_raytracing.h).
	// Scalars, not vec3: std430 vec3 would force 16-byte alignment and shift
	// every offset after it.
	float transmission; // 0 = opaque, 1 = full delta dielectric transmission.
	float ior;
	float absorption_r; // Beer-Lambert sigma/m per channel.
	float absorption_g;
	float absorption_b;
	// uv1 triplanar (former padding; struct size unchanged).
	// Flag bits 8 (triplanar) / 16 (world triplanar) in `flags` gate use.
	float triplanar_sharpness;
	float uv1_scale_z; // z of Godot's Vector3 uv1_scale (xy above).
	float uv1_offset_z;
	// pt_glow contract (flag bit 32 gates use). Scalars, not vec3 — same
	// std430 alignment reasoning as the transmission contract above.
	// Snapshotted host-side from the declaring shader's uniforms; LINEAR.
	// Consumed exclusively through pt_glow_evaluate below — hit path, NEE
	// and emitter-table power all share that one formula.
	float glow_color_r;
	float glow_color_g;
	float glow_color_b;
	float glow_energy;
	float glow_rim_power;
	float glow_rim_boost;
	float glow_core_floor;
	float glow_membrane_boost;
	float glow_pulse_rate_hz;
	float glow_pulse_depth;
	float _pad_glow0;
	float _pad_glow1;
};

// Material flag bits (must match RT_MAT_FLAG_* in render_raytracing.h).
const uint RT_MAT_FLAG_PT_GLOW = 32u;

// ============================================================================
// pt_glow — THE canonical organism glow evaluation
// (HANDOFF_DIFFUSE_EMISSIVES Step 1). The raster shader (organism_flesh)
// computes the same formula from the same uniforms for renderer parity, but
// under PT this function is authoritative: the closest-hit override, the NEE
// triangle sampler (Step 2) and the host emitter-table power integration
// (which replicates it in C++) must all agree with THIS text.
//
// Inputs are the baked attributes the fork's hit fetch actually reads:
//   vertex_color: COLOR — .b edge distance, .a emission mask (.r/.g are
//                 palette weights, unused by glow)
//   uv:           UV    — .x growth (unused by glow), .y membrane
// pulse phase is derived from the instance's world origin, matching the
// raster shader's vertex-stage hash — same instance, same phase, both
// renderers.
// NOTE: exposure normalization (scene_data.emissive_exposure_normalization)
// is applied by the CALLER, matching how the custom-fragment emission path
// applies it.
// ============================================================================
float pt_glow_phase(vec3 instance_world_origin) {
	return fract(sin(dot(instance_world_origin, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
}

vec3 pt_glow_evaluate(MaterialData md, vec4 vertex_color, vec2 uv, float time_s, float phase) {
	float edge_dist = vertex_color.b;
	float membrane = uv.y;
	float emission_mask = vertex_color.a;
	float rim = pow(clamp(1.0 - edge_dist, 0.0, 1.0), md.glow_rim_power);
	float glow = (md.glow_core_floor + rim * md.glow_rim_boost + membrane * md.glow_membrane_boost) * emission_mask;
	float pulse = 1.0 + md.glow_pulse_depth * sin(time_s * 6.28318530718 * md.glow_pulse_rate_hz + phase * 6.28318530718);
	return vec3(md.glow_color_r, md.glow_color_g, md.glow_color_b) * (md.glow_energy * glow * pulse);
}
