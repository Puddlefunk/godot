// Common closest_hit utilities shared by all hit groups.
//
// Required includes (before this file):
//   raytracing_inc.glsl, brdf_inc.glsl, raytracing_hit_inc.glsl, raytracing_lights_inc.glsl,
//   raytracing_material_eval_inc.glsl
//
// Required bindings (before this file):
//   tlas, payload, scene_data_block, geometries[], motion_indices[], materials[], motion_transforms[], bindless_textures[],
//   SAMPLER_* (12 material samplers), rt_params, rt_depth_image,
//   DLSS-RR images (ifdef DLSS_RR_ENABLED)

// ============================================================================
// HIT DATA
// ============================================================================

struct HitData {
	vec3 hit_pos;
	vec3 geometry_normal; // World space, flipped for back-face hits.
	vec3 tangent; // World space.
	vec3 bitangent; // World space.
	vec2 uv; // Raw UV (no material scale/offset applied).
	vec4 color; // Vertex color (white if not present).
	bool is_front_face;
	uint geometry_idx;
};

/// Fetch vertex attributes and transform to world space.
/// Requires hitAttributeEXT HitAttribs and GeometryBuffer/MaterialBuffer bindings.
HitData compute_hit_data() {
	HitData h;
	h.geometry_idx = gl_InstanceCustomIndexEXT;
	GeometryData geom = geometries[h.geometry_idx];

	// Custom hit groups may reference vertex color in their fragment code, so
	// pull all attributes; default HGs can skip color for perf.
#ifdef RT_CUSTOM_HIT_GROUP
	VertexAttributes attrs = fetch_vertex_attributes(geom, attribs, FETCH_ALL);
#else
	VertexAttributes attrs = fetch_vertex_attributes(geom, attribs, FETCH_UV | FETCH_TBN);
#endif
	h.uv = attrs.uv;
	h.color = attrs.color;

	mat3 model_rotation = mat3(gl_ObjectToWorldEXT);
	mat3 normal_matrix = mat3(
			normalize(model_rotation[0]),
			normalize(model_rotation[1]),
			normalize(model_rotation[2]));

#ifdef ENABLE_INTERSECTION_SHADERS
	if ((geom.flags & FLAG_PROCEDURAL) != 0u) {
		h.uv = hit_attribs.bary_or_uv;
		vec3 obj_normal = normalize(unpackSnorm4x8(hit_attribs.packed_normal).xyz);
		vec3 obj_tangent = normalize(unpackSnorm4x8(hit_attribs.packed_tangent).xyz);
		h.geometry_normal = normalize(normal_matrix * obj_normal);
		h.tangent = normalize(normal_matrix * obj_tangent);
		h.bitangent = cross(h.geometry_normal, h.tangent);

		h.is_front_face = (dot(h.geometry_normal, -gl_WorldRayDirectionEXT) > 0.0);
		if (!h.is_front_face) {
			h.geometry_normal = -h.geometry_normal;
		}
	} else
#endif
	{
		// Triangle hit: reuse `attrs` from the top-level fetch (already has
		// UV / TBN from FETCH_UV | FETCH_TBN or FETCH_ALL).
		h.geometry_normal = normalize(normal_matrix * attrs.normal);
		h.tangent = normalize(normal_matrix * attrs.tangent);
		h.bitangent = cross(h.geometry_normal, h.tangent) * attrs.bitangent_sign;

		h.is_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
		if (!h.is_front_face) {
			h.geometry_normal = -h.geometry_normal;
		}
	}

	h.hit_pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

	return h;
}

// ============================================================================
// HELPERS
// ============================================================================

vec4 sample_bindless_texture(uint tex_idx, vec2 uv) {
	return texture(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv);
}

/// Sample with point/nearest filtering (for pixel art textures).
vec4 sample_bindless_texture_point(uint tex_idx, vec2 uv) {
	return texture(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_REPEAT), uv);
}

/// Sample with the appropriate filter based on material flags (bit 2 = point filtering).
vec4 sample_material_texture(uint tex_idx, vec2 uv, uint mat_flags) {
	if ((mat_flags & 4u) != 0u) {
		return sample_bindless_texture_point(tex_idx, uv);
	}
	return sample_bindless_texture(tex_idx, uv);
}

/// Select one scalar from a sampled texel by BaseMaterial3D TextureChannel index
/// (0=R 1=G 2=B 3=A 4=grayscale). Raster does this with a vec4 dot-mask
/// (roughness_texture_channel etc.); the host packs the mask down to this index
/// in MaterialData.texture_channels.
float rt_channel_sample(vec4 s, uint ch) {
	if (ch == 0u) {
		return s.r;
	}
	if (ch == 1u) {
		return s.g;
	}
	if (ch == 2u) {
		return s.b;
	}
	if (ch == 3u) {
		return s.a;
	}
	return dot(s.rgb, vec3(1.0 / 3.0));
}

// ============================================================================
// TEXTURE LOD (first-order ray cone)
// ============================================================================
// Ray/hit stages have no implicit derivatives: `texture()` compiles to an
// explicit LOD-0 sample in SPIR-V, so every material texture used to be read
// at full resolution regardless of hit distance. Raster mip-filters
// high-frequency normal maps toward flat exactly where triplanar projections
// stretch (the near-tangent axes); at LOD 0 the PT hit path read raw
// full-contrast texels there, and the violently perturbed normals became the
// bright "shiny splits/stretches" streaks on normal-mapped rock — PT-only,
// scaling with normal_scale, invisible to any math-level audit because the
// formulas DID mirror raster. Sampling at the ray-cone footprint restores
// raster's filtering, which is what lets a heavy normal_scale look heavy
// without fireflies.
//
// First-order cone: footprint = hit distance * angular size of one pixel.
// Segment-local (gl_HitTEXT, not accumulated path length): the payload has no
// spare bits for cone state (PAYLOAD_BUDGET.md), and primary hits are where
// the artifact lives; secondary bounces only under-filter, and they are
// denoised anyway.
float rt_cone_width_at_hit() {
	// projection_matrix[1][1] == 1/tan(fov_y/2) for a perspective projection;
	// one pixel subtends ~2*tan(fov_y/2)/viewport_h radians at the image
	// center. (Orthographic dev cameras get a slight over-blur — acceptable.)
	float pixel_spread = 2.0 / (scene_data_block.data.projection_matrix[1][1] * scene_data_block.data.viewport_size.y);
	return gl_HitTEXT * pixel_spread;
}

/// One vertex position in BLAS OBJECT space, for both vertex layouts -- i.e.
/// exactly the space `gl_ObjectToWorldEXT` expects, in both cases.
///
/// Compressed surfaces store position as three 16-bit unorms normalised into
/// the surface AABB (position_stride is 4x uint16: x, y, z, then the tangent
/// angle that fetch_tbn_compressed_vertex reads out of the high half of the
/// second word). The acceleration structure was built from those unorms
/// directly, so for compressed geometry BLAS object space IS the unit cube and
/// the AABB decompression is baked into the instance transform -- which is why
/// the model matrix has to be recovered as `gl_ObjectToWorldEXT *
/// inv_aabb_xform` (raytracing_custom_fragment_inc, and compute_triplanar_coords
/// below goes the other way with `aabb_xform * object_hit_pos`).
///
/// So this deliberately does NOT apply aabb_xform: applying it and then
/// gl_ObjectToWorldEXT would count the AABB scale twice and inflate world areas
/// by that factor squared.
vec3 rt_fetch_object_position(in GeometryData geom, uint idx) {
	if ((geom.flags & FLAG_COMPRESSED) != 0u) {
		Uint32Buffer vb = Uint32Buffer(geom.vertex_address);
		uint w0 = vb.v[idx * 2u + 0u];
		uint w1 = vb.v[idx * 2u + 1u];
		return vec3(float(w0 & 0xFFFFu), float(w0 >> 16), float(w1 & 0xFFFFu)) / 65535.0;
	}
	FloatBuffer vb_f = FloatBuffer(geom.vertex_address);
	uint stride_floats = geom.position_stride >> 2;
	return vec3(vb_f.v[idx * stride_floats + 0u],
			vb_f.v[idx * stride_floats + 1u],
			vb_f.v[idx * stride_floats + 2u]);
}

/// UV units per world unit for the CURRENT triangle, or 0 when undeterminable.
///
/// THE FOLLOW-UP FIXES-2026-08-17 WROTE DOWN. That session gave triplanar
/// sampling a ray-cone LOD and deliberately left mesh-UV sampling at LOD 0,
/// because "texel density per world unit is unknowable without per-triangle UV
/// area, and no artifact has been observed there (marble is clean)" -- with the
/// note that distant mesh-UV materials sparkling would be the trigger to fix
/// it. Alpha-cut foliage is the first mesh-UV content in the project that is
/// thin, high-contrast and viewed at range, and it sparkles: a 2k atlas read at
/// LOD 0 across sub-pixel quads lands on a different texel every time the
/// camera twitches.
///
/// The area is obtainable right here. The hit shader already has the three
/// indices; their UVs and positions are two buffer reads away.
///
/// WORLD area, not object area: instances are scaled -- MultiMesh foliage
/// especially, where every blade carries its own random scale -- so object
/// space would be wrong by whatever that scale happens to be.
///
/// Returns 0 rather than guessing whenever the inputs are degenerate, and 0
/// means "stay at LOD 0", i.e. exactly the behaviour before this existed.
float rt_mesh_uv_density(in GeometryData geom, uint i0, uint i1, uint i2) {
	if (geom.vertex_address == 0ul || geom.attribute_address == 0ul) {
		return 0.0;
	}
	// 2D surfaces store two floats per position; there is no triangle area to
	// speak of and the third read would be another vertex's x.
	if ((geom.flags & FLAG_COMPRESSED) == 0u && geom.position_stride < 12u) {
		return 0.0;
	}

	// Barycentric selectors pull the three corner UVs through the existing
	// decoder rather than duplicating its compressed/uncompressed branch here.
	vec2 uv0 = fetch_uv(geom, i0, i1, i2, vec3(1.0, 0.0, 0.0));
	vec2 uv1 = fetch_uv(geom, i0, i1, i2, vec3(0.0, 1.0, 0.0));
	vec2 uv2 = fetch_uv(geom, i0, i1, i2, vec3(0.0, 0.0, 1.0));
	float uv_area = abs((uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv2.x - uv0.x) * (uv1.y - uv0.y));
	if (uv_area <= 1e-12) {
		return 0.0;
	}

	vec3 p0 = vec3(gl_ObjectToWorldEXT * vec4(rt_fetch_object_position(geom, i0), 1.0));
	vec3 p1 = vec3(gl_ObjectToWorldEXT * vec4(rt_fetch_object_position(geom, i1), 1.0));
	vec3 p2 = vec3(gl_ObjectToWorldEXT * vec4(rt_fetch_object_position(geom, i2), 1.0));
	float world_area = length(cross(p1 - p0, p2 - p0));
	if (world_area <= 1e-12) {
		return 0.0;
	}

	float density = sqrt(uv_area / world_area);
	return (isinf(density) || isnan(density)) ? 0.0 : density;
}

/// Mip level for a mesh-UV texture read, mirroring the triplanar path's
/// footprint-times-texture-size formula so both take the same shape.
float rt_mesh_uv_lod(uint tex_idx, float density) {
	if (density <= 0.0) {
		return 0.0;
	}
	vec2 tex_size = vec2(textureSize(sampler2D(bindless_textures[nonuniformEXT(tex_idx)],
			SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), 0));
	float texels = rt_cone_width_at_hit() * density * max(tex_size.x, tex_size.y);
	return clamp(log2(max(texels, 1.0)), 0.0, 15.0);
}

/// sample_material_texture with an explicit LOD. Point-filtered (pixel art)
/// materials keep nearest filtering within the selected mip.
vec4 sample_material_texture_lod(uint tex_idx, vec2 uv, uint mat_flags, float lod) {
	if ((mat_flags & 4u) != 0u) {
		return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_NEAREST_WITH_MIPMAPS_REPEAT), uv, lod);
	}
	return textureLod(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv, lod);
}

/// Apply tangent-space normal map to geometry normal.
vec3 apply_normal_map(HitData h, vec3 tangent_space_normal, float normal_map_depth) {
	vec3 mapped = h.tangent * tangent_space_normal.x + h.bitangent * tangent_space_normal.y + h.geometry_normal * tangent_space_normal.z;
	return normalize(mix(h.geometry_normal, mapped, normal_map_depth));
}

// ============================================================================
// UV1 TRIPLANAR (StandardMaterial3D parity under PT)
// ============================================================================
// Raster triplanar lives in BaseMaterial3D's GENERATED shader code, which the
// PT hit path never runs -- so PT reimplements the same math from the record:
// weights = pow(abs(N), sharpness)/sum, pos = P * uv1_scale + uv1_offset then
// pos *= (1,-1,1), samples xy*w.z + xz*w.y + zy*(-1,1)*w.x (material.cpp
// ~1391-1473 is the reference). Object mode uses MODEL space: for compressed
// meshes BLAS object space is the unit-cube compression space, so positions
// go through aabb_xform and normals through the model matrix transpose
// (same construction as read_model_matrix in the custom-fragment path).

struct TriplanarCoords {
	vec3 weights;
	vec3 pos;
	vec3 tangent; // World-space axis-aligned frame for normal mapping,
	vec3 binormal; // matching raster's generated TANGENT/BINORMAL.
	// Per-axis UV-space footprint of the ray cone at this hit. Each projection
	// axis stretches by 1/|n_axis| (surface near-parallel to the projection ->
	// long smeared texel runs -> the streak zones), so each axis carries its
	// own footprint. Multiplied by the texture's size at sample time to get a
	// per-axis LOD (see sample_material_texture_triplanar).
	vec3 uv_width;
};

TriplanarCoords compute_triplanar_coords(HitData h, MaterialData mat, GeometryData geom) {
	TriplanarCoords tc;
	bool world_mode = (mat.flags & 16u) != 0u;
	vec3 n; // Blend-space normal (world or model).
	vec3 p; // Blend-space position.
	mat4 model = mat4(1.0);
	if (world_mode) {
		n = h.geometry_normal;
		p = h.hit_pos;
	} else {
		mat4 aabb_xform, inv_aabb_xform;
		get_aabb_compression_xforms(geom, aabb_xform, inv_aabb_xform);
		model = mat4(gl_ObjectToWorldEXT) * inv_aabb_xform;
		// n_model = transpose(M) * n_world (inverse of the inverse-transpose).
		n = normalize((vec4(h.geometry_normal, 0.0) * model).xyz);
		p = (aabb_xform * vec4(gl_ObjectRayOriginEXT + gl_HitTEXT * gl_ObjectRayDirectionEXT, 1.0)).xyz;
	}

	tc.weights = pow(abs(n), vec3(mat.triplanar_sharpness));
	tc.weights /= dot(tc.weights, vec3(1.0));

	p = p * vec3(mat.uv1_scale, mat.uv1_scale_z) + vec3(mat.uv1_offset, mat.uv1_offset_z);
	tc.pos = p * vec3(1.0, -1.0, 1.0);

	// Raster's axis-aligned triplanar frame (material.cpp ~1369-1388),
	// brought to world space for object mode.
	vec3 t = vec3(0.0, 0.0, -1.0) * abs(n.x) + vec3(1.0, 0.0, 0.0) * abs(n.y) + vec3(1.0, 0.0, 0.0) * abs(n.z);
	vec3 b = vec3(0.0, 1.0, 0.0) * abs(n.x) + vec3(0.0, 0.0, -1.0) * abs(n.y) + vec3(0.0, 1.0, 0.0) * abs(n.z);
	if (!world_mode) {
		t = mat3(model) * t;
		b = mat3(model) * b;
	}
	tc.tangent = normalize(t);
	tc.binormal = normalize(b);

	// Ray-cone footprint per projection axis, in UV units. Axis i samples a
	// plane the cone crosses at |n_i| incidence: footprint stretches by
	// 1/|n_i|, unbounded as the surface goes tangent to the projection —
	// exactly where raster's derivative-driven mips averaged the normal map
	// flat and LOD-0 PT did not. The 1e-3 floor only bounds the arithmetic;
	// hardware clamps to the last mip anyway. UV scale per axis pairs the two
	// scale components that projection actually uses (max = conservative, like
	// trilinear picking the larger derivative).
	float cone_width = rt_cone_width_at_hit();
	vec3 axis_scale = vec3(
			max(abs(mat.uv1_scale_z), abs(mat.uv1_scale.y)), // x axis samples pos.zy
			max(abs(mat.uv1_scale.x), abs(mat.uv1_scale_z)), // y axis samples pos.xz
			max(abs(mat.uv1_scale.x), abs(mat.uv1_scale.y))); // z axis samples pos.xy
	tc.uv_width = cone_width * axis_scale / max(abs(n), vec3(1e-3));
	return tc;
}

vec4 sample_material_texture_triplanar(uint tex_idx, TriplanarCoords tc, uint mat_flags) {
	// Per-axis LOD from the cone footprint and this texture's resolution.
	// textureSize is per-texture, so this cannot live in TriplanarCoords.
	float tex_size = float(textureSize(sampler2D(bindless_textures[nonuniformEXT(tex_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), 0).y);
	vec3 lods = log2(max(tc.uv_width * tex_size, vec3(1.0)));
	vec4 samp = vec4(0.0);
	samp += sample_material_texture_lod(tex_idx, tc.pos.xy, mat_flags, lods.z) * tc.weights.z;
	samp += sample_material_texture_lod(tex_idx, tc.pos.xz, mat_flags, lods.y) * tc.weights.y;
	samp += sample_material_texture_lod(tex_idx, tc.pos.zy * vec2(-1.0, 1.0), mat_flags, lods.x) * tc.weights.x;
	return samp;
}

/// Normal mapping over the triplanar frame instead of mesh tangents.
vec3 apply_normal_map_triplanar(HitData h, TriplanarCoords tc, vec3 tangent_space_normal, float normal_map_depth) {
	vec3 mapped = tc.tangent * tangent_space_normal.x + tc.binormal * tangent_space_normal.y + h.geometry_normal * tangent_space_normal.z;
	return normalize(mix(h.geometry_normal, mapped, normal_map_depth));
}

// ============================================================================
// DEPTH WRITE (primary ray only)
// ============================================================================

/// Write NDC depth for primary ray hits (bounce 0, sample 0 only).
void write_primary_hit_depth(vec3 hit_pos) {
	if (get_total_bounces(payload.packed_bounces_flags) == 0u && is_sample_zero(payload.packed_bounces_flags)) {
		mat4 view_mat = transpose(mat4(scene_data_block.data.view_matrix[0],
				scene_data_block.data.view_matrix[1],
				scene_data_block.data.view_matrix[2],
				vec4(0.0, 0.0, 0.0, 1.0)));
		vec3 view_pos = (view_mat * vec4(hit_pos, 1.0)).xyz;
		vec4 clip_pos = scene_data_block.data.projection_matrix * vec4(view_pos, 1.0);
		float ndc_depth = clip_pos.z / clip_pos.w;
		imageStore(rt_depth_image, ivec2(gl_LaunchIDEXT.xy), vec4(ndc_depth));
	}
}

// ============================================================================
// VELOCITY WRITE (primary ray only, MV-gated)
// ============================================================================

#ifdef ENABLE_INTERSECTION_SHADERS
/// Decode the FP16-compressed PREV_POSITION delta from HitAttribs.
vec3 decode_prev_pos_delta() {
	uint dx_low = (hit_attribs.packed_normal >> 24u) & 0xFFu;
	uint dx_high = (hit_attribs.packed_tangent >> 24u) & 0xFFu;
	float delta_x = unpackHalf2x16(dx_low | (dx_high << 8u)).x;
	vec2 delta_yz = unpackHalf2x16(hit_attribs.prev_pos_delta_yz);
	return vec3(delta_x, delta_yz.x, delta_yz.y);
}
#endif

/// Reconstruct previous-frame mat4 from a compact motion transform entry.
mat4 decode_prev_object_to_world(int motion_idx) {
	InstanceMotionData m = motion_transforms[motion_idx];
	return transpose(mat4(
			vec4(m.prev_xform[0], m.prev_xform[1], m.prev_xform[2], m.prev_xform[3]),
			vec4(m.prev_xform[4], m.prev_xform[5], m.prev_xform[6], m.prev_xform[7]),
			vec4(m.prev_xform[8], m.prev_xform[9], m.prev_xform[10], m.prev_xform[11]),
			vec4(0.0, 0.0, 0.0, 1.0)));
}

/// Write motion vectors for primary ray hits (bounce 0, sample 0 only).
/// Uses unjittered VP matrices matching the raster motion_vectors_store convention.
void write_primary_hit_velocity(vec3 hit_pos) {
	if (get_total_bounces(payload.packed_bounces_flags) != 0u || !is_sample_zero(payload.packed_bounces_flags)) {
		return;
	}

	uint geom_idx = gl_InstanceCustomIndexEXT;
	int mi = motion_indices[geom_idx];

	// Resolve previous-frame model matrix: compact entry if moved, current transform otherwise.
	mat4 prev_model = (mi >= 0) ? decode_prev_object_to_world(mi) : mat4(gl_ObjectToWorldEXT);

	vec3 obj_pos = (mat4(gl_WorldToObjectEXT) * vec4(hit_pos, 1.0)).xyz;
	vec3 prev_obj_pos = obj_pos;

	GeometryData geom = geometries[geom_idx];

#ifdef ENABLE_INTERSECTION_SHADERS
	if ((geom.flags & FLAG_PROCEDURAL) != 0u) {
		prev_obj_pos += decode_prev_pos_delta();
	}
#endif

#ifdef RT_HIT_ATTRIBS_DECLARED
	if ((geom.flags & FLAG_DEFORMED) != 0u) {
		uint64_t prev_addr = packUint2x32(uvec2(geom.prev_vertex_address_lo, geom.prev_vertex_address_hi));
		if (prev_addr != 0ul) {
			uint i0, i1, i2;
			get_triangle_indices(geom, i0, i1, i2);
			vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
			FloatBuffer prev_vb = FloatBuffer(prev_addr);
			uint stride_floats = geom.position_stride >> 2;
			vec3 p0 = vec3(prev_vb.v[i0 * stride_floats + 0u],
					prev_vb.v[i0 * stride_floats + 1u],
					prev_vb.v[i0 * stride_floats + 2u]);
			vec3 p1 = vec3(prev_vb.v[i1 * stride_floats + 0u],
					prev_vb.v[i1 * stride_floats + 1u],
					prev_vb.v[i1 * stride_floats + 2u]);
			vec3 p2 = vec3(prev_vb.v[i2 * stride_floats + 0u],
					prev_vb.v[i2 * stride_floats + 1u],
					prev_vb.v[i2 * stride_floats + 2u]);
			prev_obj_pos = bary.x * p0 + bary.y * p1 + bary.z * p2;
		}
	}
#endif

	vec3 prev_world_pos = (prev_model * vec4(prev_obj_pos, 1.0)).xyz;

	vec2 curr_uv = project_uv(hit_pos, curr_vp_unjittered);
	vec2 prev_uv = project_uv(prev_world_pos, prev_vp_unjittered);

	imageStore(rt_velocity_image, ivec2(gl_LaunchIDEXT.xy), vec4(prev_uv - curr_uv, 0.0, 0.0));
}

// ============================================================================
// ENVIRONMENT FOG (per ray segment)
// ============================================================================

vec3 fog_get_directional_color(uint index) {
	return rt_lights[index].emission;
}

vec3 fog_get_directional_direction(uint index) {
	vec3 world_dir = -normalize(rt_lights[index].position);
	mat3 view_rot = transpose(mat3(
			scene_data_block.data.view_matrix[0].xyz,
			scene_data_block.data.view_matrix[1].xyz,
			scene_data_block.data.view_matrix[2].xyz));
	return view_rot * world_dir;
}

#define FOG_HAS_RADIANCE

vec3 fog_sample_radiance(vec3 vertex, float mip_level) {
	vec3 cube_view = scene_data_block.data.radiance_inverse_xform * vertex;
	vec2 border = vec2(scene_data_block.data.radiance_border_size,
			1.0 - scene_data_block.data.radiance_border_size * 2.0);
	vec2 cube_uv = vec3_to_oct_with_border(cube_view, border);
	// mip_level is a normalized roughness (0..1); radiance_octmap_sample maps it
	// onto the prefiltered roughness array layers.
	return radiance_octmap_sample(cube_uv, mip_level);
}

#include "../fog_inc.glsl"

/// Apply environment fog for the ray segment that was just traversed.
/// Attenuates throughput and adds in-scattered fog color.
void apply_segment_fog(float segment_dist, inout vec3 radiance, inout vec3 throughput) {
	if ((RT_FLAGS & RT_FLAG_FOG_ENABLED) == 0u) {
		return;
	}

	// Build a view-space vertex along the ray direction at the hit distance.
	// fog_process needs view-space position for distance and height calculations.
	mat4 view_mat = transpose(mat4(
			scene_data_block.data.view_matrix[0],
			scene_data_block.data.view_matrix[1],
			scene_data_block.data.view_matrix[2],
			vec4(0.0, 0.0, 0.0, 1.0)));
	vec3 world_hit = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * segment_dist;
	vec3 vertex = (view_mat * vec4(world_hit, 1.0)).xyz;

	vec4 fog = fog_process(scene_data_block.data, vertex);
	radiance += throughput * fog.rgb * fog.a;
	throughput *= (1.0 - fog.a);
}

/// Converts specular parameter [0..1] to dielectric F0.
float specular_to_f0(float specular) {
	return 0.16 * specular * specular;
}

// ============================================================================
// DEBUG VISUALIZATION
// ============================================================================

#ifdef RT_DEBUG_ENABLED
void debug_visualize(
		int vis_mode,
		vec3 geometry_normal,
		vec3 final_normal,
		vec3 tangent_space_normal,
		vec3 world_tangent,
		vec3 world_bitangent,
		vec2 uv,
		vec3 albedo,
		vec3 orm,
		float metalness,
		float roughness,
		float specular,
		vec3 emissive,
		vec3 V,
		float NdotV) {
	PathState ps = path_unpack(payload);

	if (vis_mode == 1) {
		if (get_total_bounces(ps.packed_bounces_flags) == 0u) {
			ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
			ps.hit_t = gl_HitTEXT;
			ps.offset_normal = geometry_normal;
			ps.next_ray_dir = reflect(gl_WorldRayDirectionEXT, geometry_normal);
			path_pack(payload, ps);
			return;
		} else {
			ps.radiance = geometry_normal * 0.5 + 0.5;
		}
	} else if (vis_mode == 2) {
		ps.radiance = geometry_normal * 0.5 + 0.5;
	} else if (vis_mode == 3) {
		ps.radiance = final_normal * 0.5 + 0.5;
	} else if (vis_mode == 4) {
		ps.radiance = tangent_space_normal * 0.5 + 0.5;
	} else if (vis_mode == 5) {
		ps.radiance = world_tangent * 0.5 + 0.5;
	} else if (vis_mode == 6) {
		ps.radiance = world_bitangent * 0.5 + 0.5;
	} else if (vis_mode == 7) {
		ps.radiance = vec3(fract(uv), 0.0);
	} else if (vis_mode == 8) {
		ps.radiance = albedo;
	} else if (vis_mode == 9) {
		ps.radiance = orm;
	} else if (vis_mode == 10) {
		ps.radiance = DLSSRR_computeDiffuseAlbedo(albedo, metalness);
	} else if (vis_mode == 11) {
		ps.radiance = DLSSRR_computeSpecularAlbedo(albedo, metalness, specular_to_f0(specular), roughness, NdotV);
	} else if (vis_mode == 12) {
		ps.radiance = (final_normal * 0.5 + 0.5) * (1.0 - roughness * 0.5);
	} else if (vis_mode == 13) {
		if (get_total_bounces(ps.packed_bounces_flags) == 0u) {
			if (roughness < MAX_DENOISER_SPECULAR_HIT_THRESHOLD) {
				ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
				ps.radiance = vec3(0.1, 0.1, 0.4);
				ps.hit_t = gl_HitTEXT;
				ps.offset_normal = final_normal;
				ps.next_ray_dir = reflect(gl_WorldRayDirectionEXT, final_normal);
				path_pack(payload, ps);
				return;
			} else {
				ps.radiance = vec3(0.1, 0.1, 0.4);
			}
		} else {
			float spec_hit_t = gl_HitTEXT;
			float v = clamp(log(spec_hit_t + 1.0) / log(1000.0), 0.0, 1.0);
			vec3 color;
			if (v < 0.33) {
				color = mix(vec3(0.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0), v * 3.0);
			} else if (v < 0.66) {
				color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 0.0), (v - 0.33) * 3.0);
			} else {
				color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), (v - 0.66) * 3.0);
			}
			ps.radiance = color;
		}
	} else if (vis_mode == 14) {
		ps.radiance = vec3(metalness);
	} else if (vis_mode == 15) {
		ps.radiance = vec3(roughness);
	} else if (vis_mode == 16) {
		mat3 world_to_view = mat3(scene_data_block.data.inv_view_matrix);
		ps.radiance = normalize(world_to_view * final_normal) * 0.5 + 0.5;
	} else if (vis_mode == 17) {
		vec3 diffuse_albedo = DLSSRR_computeDiffuseAlbedo(albedo, metalness);
		vec3 specular_albedo = DLSSRR_computeSpecularAlbedo(albedo, metalness, specular_to_f0(specular), roughness, NdotV);
		ps.radiance = mix(diffuse_albedo, specular_albedo, metalness);
	} else if (vis_mode == 18) {
		ps.radiance = baseColorToSpecularF0(albedo, metalness, specular_to_f0(specular));
	} else if (vis_mode == 19) {
		bool is_front_face = (gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT);
		ps.radiance = is_front_face ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
	} else if (vis_mode == 20) {
		float depth_range = scene_data_block.data.z_far - scene_data_block.data.z_near;
		float d = clamp(gl_HitTEXT / depth_range, 0.0, 1.0);
		ps.radiance = vec3(d);
	} else if (vis_mode == 21) {
		ps.radiance = emissive;
	} else if (vis_mode == 22) {
		// BRDF below-hemisphere fallback visualization. Reflects the two-layer
		// recovery applied in shade_and_bounce:
		//   Layer 1 - shading-normal clamp toward geometry (energy preserving
		//             but flattens detail at grazing angles)
		//   Layer 2 - mirror rejected directions across the geometry plane
		//             (biased; reuses the rejected sample's BRDF weight)
		//
		// We sample both BRDF lobes once with the un-clamped shading normal
		// and once with the clamped shading normal, reusing the same random
		// pair so the two passes are directly comparable.
		//
		// Color legend (ordered cleanest -> most biased):
		//   green   = no rejection in either pass (no fallback needed)
		//   blue    = clamp alone resolves rejection (energy preserving)
		//   yellow  = clamp + mirror (one lobe still mirrored after clamp)
		//   red     = clamp + mirror (both lobes mirrored - most biased pixel)
		//
		// Production output is never black anymore -- yellow/red just indicate
		// where the cheaper Layer-1 clamp could not catch the bump and the
		// biased Layer-2 mirror had to step in. Raise
		// RT_SHADING_NORMAL_CLAMP_THRESHOLD if yellow/red dominate.
		MaterialProperties dbg_mat;
		dbg_mat.baseColor = albedo;
		dbg_mat.metalness = metalness;
		dbg_mat.roughness = roughness;
		dbg_mat.dielectricF0 = specular_to_f0(specular);
		dbg_mat.emissive = vec3(0.0);
		dbg_mat.transmissivness = 0.0;
		dbg_mat.eta = 1.0;
		dbg_mat.opacity = 1.0;
		dbg_mat.diffuseTransmission = vec3(0.0);

		vec3 dbg_dir;
		vec3 dbg_weight;
		// Draw both random pairs up front so each (specular/diffuse) test in
		// the un-clamped and clamped pass uses the exact same numbers.
		vec2 u_spec_rng = rand2(ps.rng_state);
		vec2 u_diff_rng = rand2(ps.rng_state);

		bool u_spec_ok = evalIndirectCombinedBRDF(u_spec_rng, final_normal, geometry_normal, V, dbg_mat, SPECULAR_TYPE, dbg_dir, dbg_weight, vec4(0.0));
		bool u_diff_ok = evalIndirectCombinedBRDF(u_diff_rng, final_normal, geometry_normal, V, dbg_mat, DIFFUSE_TYPE, dbg_dir, dbg_weight, vec4(0.0));

		// Uses the same threshold as shade_and_bounce so the visualization
		// stays in sync with production sampling.
		vec3 N_clamped = clampShadingNormal(final_normal, geometry_normal, V, RT_SHADING_NORMAL_CLAMP_THRESHOLD);
		bool c_spec_ok = evalIndirectCombinedBRDF(u_spec_rng, N_clamped, geometry_normal, V, dbg_mat, SPECULAR_TYPE, dbg_dir, dbg_weight, vec4(0.0));
		bool c_diff_ok = evalIndirectCombinedBRDF(u_diff_rng, N_clamped, geometry_normal, V, dbg_mat, DIFFUSE_TYPE, dbg_dir, dbg_weight, vec4(0.0));

		bool all_uncl_ok = u_spec_ok && u_diff_ok;
		bool all_cl_ok = c_spec_ok && c_diff_ok;
		bool any_cl_ok = c_spec_ok || c_diff_ok;

		if (all_uncl_ok) {
			ps.radiance = vec3(0.0, 1.0, 0.0);
		} else if (all_cl_ok) {
			ps.radiance = vec3(0.0, 0.4, 1.0);
		} else if (any_cl_ok) {
			ps.radiance = vec3(1.0, 1.0, 0.0);
		} else {
			ps.radiance = vec3(1.0, 0.0, 0.0);
		}
	}

	ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
	path_pack(payload, ps);
}
#endif // RT_DEBUG_ENABLED

// ============================================================================
// SHADE AND BOUNCE
// ============================================================================

/// Production shading: emissive + NEE direct lighting + BRDF importance sampling + next bounce.
/// Also handles DLSS-RR G-buffer output on primary ray.
void shade_and_bounce(HitData h, MaterialResult m) {
	PathState ps = path_unpack(payload);

	vec3 V = -gl_WorldRayDirectionEXT;

	// Clamp shading normal toward geometry at grazing view angles to keep BRDF above the geometry hemisphere.
	vec3 N = clampShadingNormal(m.normal, h.geometry_normal, V, RT_SHADING_NORMAL_CLAMP_THRESHOLD);
	float NdotV = max(dot(N, V), 0.0001);

	uint total_bounces = get_total_bounces(ps.packed_bounces_flags);
	uint diffuse_bounces = get_diffuse_bounces(ps.packed_bounces_flags);

	// Environment fog for this ray segment (before surface contribution).
	apply_segment_fog(gl_HitTEXT, ps.radiance, ps.throughput);

	// Pathtracer material contract, snapshotted from declared pt_* uniforms by
	// render_raytracing.cpp (read by name -- never inferred from raster channels).
	MaterialData contract_md = materials[h.geometry_idx];

	// pt_glow contract: the engine-owned glow formula is authoritative under
	// PT. Overrides whatever the custom fragment wrote to EMISSION so that
	// this hit, the NEE triangle sampler (Step 2) and the emitter-table
	// power (Step 1) evaluate ONE formula from the same snapshotted params —
	// the two-copies-in-sync problem structurally cannot exist.
	// Phase is per-instance, hashed CPU-side into this instance's materials[]
	// copy (see glow_pulse_phase in raytracing_data_inc.glsl).
	if ((contract_md.flags & RT_MAT_FLAG_PT_GLOW) != 0u) {
		m.emissive = pt_glow_evaluate(contract_md, h.color, h.uv, scene_data_block.data.time, contract_md.glow_pulse_phase) *
				scene_data_block.data.emissive_exposure_normalization;
	}

	// Beer-Lambert absorption. If we are exiting a transmissive medium
	// through its back face, the segment just traversed was INSIDE it --
	// attenuate over gl_HitTEXT meters with the precomputed sigma. CORRECT
	// ONLY FOR CONVEX SOLIDS: on a non-convex body (bell, shell) this bills
	// interior air gaps to the medium. Nested/non-convex media need interior
	// tracking -- planned, not yet built. Before the emissive add, so
	// anything seen at or beyond this surface is tinted by the interior.
	if (contract_md.transmission > 0.0 && !h.is_front_face) {
		vec3 sigma = vec3(contract_md.absorption_r, contract_md.absorption_g, contract_md.absorption_b);
		ps.throughput *= exp(-sigma * gl_HitTEXT);
	}

	// Emissive contribution, MIS-weighted against the NEE emissive-triangle
	// sampler (HANDOFF_DIFFUSE_EMISSIVES Steps 0/3). While
	// emissive_triangle_pdf is the Step-0 stub returning 0.0 the weight is
	// exactly 1.0 — bit-identical to the historical unweighted add. Camera
	// rays (total_bounces == 0) and delta/invalid previous bounces are
	// unweighted by the prev_bsdf_pdf > 0.0 gate: the light sampler can
	// never generate those path segments, so BSDF sampling keeps full
	// credit for them.
	float emissive_mis_weight = 1.0;
	float prev_bsdf_pdf = mis_pdf_load(ps.packed_bounces_flags);
	if (total_bounces > 0u && prev_bsdf_pdf > 0.0) {
		float emissive_light_pdf = emissive_triangle_pdf(h.geometry_idx, uint(gl_PrimitiveID),
				gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT, gl_HitTEXT, h.geometry_normal);
		emissive_mis_weight = prev_bsdf_pdf / (prev_bsdf_pdf + emissive_light_pdf);
	}
	ps.radiance += ps.throughput * m.emissive * emissive_mis_weight;

	// Bounce limits (Step 4). The total-bounce cap stays as the hard ceiling
	// (the payload counters' carry-free arithmetic depends on it,
	// PAYLOAD_BUDGET.md §3). The old MAX_DIFFUSE_BOUNCES 2 truncation is
	// retired: from that diffuse depth on, RUSSIAN ROULETTE decides
	// continuation. Survival probability tracks remaining throughput; the
	// survivor's throughput is divided by it, so the estimator is unbiased
	// where the cap silently deleted energy — multi-bounce diffuse
	// interiors, and every backlit-membrane path (a diffuse-transmission
	// bounce counts as diffuse). A terminated path keeps this vertex's
	// emission — added above, before the roll — and drops NEE +
	// continuation, the exact shape the old cap had, reweighted instead of
	// truncated: NEE at this vertex is only evaluated by survivors, whose
	// 1/p makes its expectation come out right.
	if (total_bounces >= RT_GET_MAX_BOUNCES()) {
		ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
		path_pack(payload, ps);
		return;
	}
	if (diffuse_bounces >= RT_RR_START_DIFFUSE_BOUNCES) {
		float rr_p = clamp(max(ps.throughput.r, max(ps.throughput.g, ps.throughput.b)), RT_RR_P_MIN, RT_RR_P_MAX);
		if (rand(ps.rng_state) >= rr_p) {
			ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
			path_pack(payload, ps);
			return;
		}
		ps.throughput /= rr_p;
	}

	// BRDF material setup.
	MaterialProperties brdf_mat;
	brdf_mat.baseColor = m.albedo;
	brdf_mat.metalness = m.metalness;
	brdf_mat.roughness = m.roughness;
	brdf_mat.dielectricF0 = specular_to_f0(m.specular);
	brdf_mat.emissive = m.emissive;
	brdf_mat.transmissivness = 0.0;
	brdf_mat.opacity = 1.0;
	brdf_mat.eta = 1.0;
	// Thin-walled diffuse transmission (Step 2b), from the shader's BACKLIGHT
	// output. Deliberately NOT contract_md.transmission: that field makes the
	// shadow any-hit pass through, and backlit flesh must keep casting shadows.
	brdf_mat.diffuseTransmission = m.backlight;

	// Consume the transmission contract.
	brdf_mat.transmissivness = contract_md.transmission;
	if (contract_md.transmission > 0.0) {
		// Delta dielectric wins outright; mixing the two transmission models
		// on one surface is not supported (and the NEE gate below assumes it).
		brdf_mat.diffuseTransmission = vec3(0.0);
		// Delta lobes only, no rough refraction -- force the record smooth so
		// the guides and the lobe agree.
		brdf_mat.roughness = 0.0;
		// Relative IOR for THIS hit: entering through the front face vs.
		// exiting through the back.
		brdf_mat.eta = h.is_front_face ? (1.0 / contract_md.ior) : contract_md.ior;
	}

	vec3 specularF0 = baseColorToSpecularF0(brdf_mat.baseColor, brdf_mat.metalness, brdf_mat.dielectricF0);
	vec3 diffuseReflectance = baseColorToDiffuseReflectance(brdf_mat.baseColor, brdf_mat.metalness);

	// =================================================================
	// DLSS Ray Reconstruction output (primary ray, sample 0 only)
	// =================================================================
#ifdef DLSS_RR_ENABLED
	if (total_bounces == 0u && is_sample_zero(ps.packed_bounces_flags)) {
		ivec2 pixel = ivec2(gl_LaunchIDEXT.xy);

		// The demodulation guides read the resolved PT material record rather than
		// the raw raster channels. These agree for ordinary surfaces, but diverge
		// wherever the record is overridden -- notably transmissive materials,
		// whose shading record is not the raster albedo.
		vec3 diffuse_albedo = DLSSRR_encodeDiffuseAlbedo(DLSSRR_computeDiffuseAlbedo(brdf_mat.baseColor, brdf_mat.metalness));

		// Emissive surfaces must carry their emission in the demodulation guide:
		// Ray Reconstruction smooths the demodulated signal as if it were lighting,
		// so detail present only in radiance is flattened. The guide is preserved
		// at full resolution, so the detail has to live here. Same approach CAPCOM
		// describe for animated emissive patterns in RE ENGINE (GDC 2026), though
		// that implementation replaces albedo outright and also covers the
		// specular guide; this only adds to diffuse.
		//
		// 11 Aug: the emission is Reinhard-normalized before it enters the
		// guide. Raw HDR emission (glow_energy beyond ~2) pushed the guide far
		// outside the range RR treats as an albedo, garbling under motion —
		// the same failure the specular guide's UNORM8 clamp below fixes
		// empirically. The guide needs the emission PATTERN, not its
		// magnitude: normalizing keeps the rim/cap/tip detail at full
		// resolution while the actual energy stays in radiance where it
		// belongs. Clamped like the specular guide for the same reason.
		vec3 guide_emissive = m.emissive;
		float guide_peak = max(guide_emissive.r, max(guide_emissive.g, guide_emissive.b));
		if (guide_peak > 0.0) {
			diffuse_albedo += guide_emissive / (1.0 + guide_peak);
		}
		diffuse_albedo = clamp(diffuse_albedo, vec3(0.0), vec3(1.0));

		imageStore(dlss_rr_diffuse_albedo, pixel, vec4(diffuse_albedo, 1.0));

		vec3 specular_albedo = DLSSRR_computeSpecularAlbedo(brdf_mat.baseColor, brdf_mat.metalness, brdf_mat.dielectricF0, brdf_mat.roughness, NdotV);
		imageStore(dlss_rr_specular_albedo, pixel, vec4(clamp(specular_albedo, vec3(0.04), vec3(1.0)), 1.0)); // match UNORM8 like before - fixes some issues with garbling..

		imageStore(dlss_rr_normal_roughness, pixel, vec4(N, brdf_mat.roughness));

		// Specular hit distance via inline ray query (only for smooth surfaces).
		float spec_hit_dist = -1.0;
		if (brdf_mat.roughness < MAX_DENOISER_SPECULAR_HIT_THRESHOLD) {
			vec3 spec_dir = reflect(-V, N);
			vec3 spec_origin = offset_ray_origin(h.hit_pos, spec_dir);

			rayQueryEXT spec_rq;
			rayQueryInitializeEXT(spec_rq, tlas, RT_RAY_FLAGS | gl_RayFlagsTerminateOnFirstHitEXT,
					RT_MASK_INDIRECT, spec_origin, 0.001, spec_dir, 10000.0);
			while (rayQueryProceedEXT(spec_rq)) {
				if (rayQueryGetIntersectionTypeEXT(spec_rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
					if (ray_query_alpha_test(
								rayQueryGetIntersectionInstanceCustomIndexEXT(spec_rq, false),
								rayQueryGetIntersectionPrimitiveIndexEXT(spec_rq, false),
								rayQueryGetIntersectionBarycentricsEXT(spec_rq, false))) {
						rayQueryConfirmIntersectionEXT(spec_rq);
					}
				}
			}
			if (rayQueryGetIntersectionTypeEXT(spec_rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
				spec_hit_dist = rayQueryGetIntersectionTEXT(spec_rq, true);
			}
		}
		imageStore(dlss_rr_specular_hit_dist, pixel, vec4(spec_hit_dist));
	}
#endif

	// =================================================================
	// NEE: Next Event Estimation (direct light sampling)
	// =================================================================
	path_pack(payload, ps);

	// Lobe-selection probability, computed ONCE and shared by the NEE-side
	// MIS weight and the BSDF sampler below — both consumers must agree or
	// the balance weights stop summing to 1.
	float specularLum = luminance(specularF0);
	float diffuseLum = luminance(diffuseReflectance);
	float specular_select_p;
	if (diffuseLum < 0.0001) {
		specular_select_p = 1.0;
	} else if (specularLum < 0.0001) {
		specular_select_p = 0.0;
	} else {
		specular_select_p = clamp(specularLum / (specularLum + diffuseLum), 0.01, 0.99);
	}

	// Low-discrepancy sequence index for CAMERA-vertex draws (Step 4 inc. B).
	// Advances per frame; at spp 2 the second sample gets its own point via
	// the sample-zero flag (the only per-sample bit the payload carries).
	// spp > 2: samples past the first fall back to white noise rather than
	// duplicating a sequence point (duplication would CORRELATE samples —
	// worse than white). Bounce vertices are always white (see the helper's
	// header note).
	uint ld_n = LD_N_INVALID;
	if (total_bounces == 0u) {
		uint ld_spp = RT_GET_SAMPLE_COUNT();
		if (ld_spp == 1u) {
			ld_n = uint(get_rt_param(RT_PARAM_FRAME_INDEX));
		} else if (ld_spp == 2u) {
			ld_n = uint(get_rt_param(RT_PARAM_FRAME_INDEX)) * 2u + (is_sample_zero(ps.packed_bounces_flags) ? 0u : 1u);
		} else if (is_sample_zero(ps.packed_bounces_flags)) {
			ld_n = uint(get_rt_param(RT_PARAM_FRAME_INDEX));
		}
	}

	uint rt_light_count = uint(get_rt_param(RT_PARAM_LIGHT_COUNT));
	uint rt_emissive_count = uint(get_rt_param(RT_PARAM_EMISSIVE_TRI_COUNT));
	// Delta transmission has zero BRDF value in any NEE-sampled light
	// direction, so skip direct light sampling entirely on those hits.
	// (Step 2b's thin-walled diffuse lobe is NEE-compatible and must NOT be
	// caught by this gate when it lands — key it on the DELTA lobe.)
	if ((rt_light_count > 0u || rt_emissive_count > 0u) && brdf_mat.transmissivness == 0.0) {
		// RAW hit position: the lights code offsets each shadow-ray origin
		// per sample along ±geometry_normal, because diffuse-transmission
		// samples (Step 2b) leave through the BACK of the surface and a
		// front-offset origin would self-occlude on the surface's own
		// triangle.
		bool is_indirect = (diffuse_bounces > 0u);
		vec3 direct_light = lights_evaluate_direct_lighting(
				h.hit_pos, h.geometry_normal, N, V, brdf_mat, ps.rng_state, is_indirect, rt_light_count,
				rt_emissive_count, get_rt_param(RT_PARAM_EMISSIVE_POOL_PROB), specular_select_p, ld_n);
		ps.radiance += ps.throughput * direct_light;
	}

	// =================================================================
	// BRDF importance sampling for next bounce
	// =================================================================
	int brdfType;
	// Thin-walled diffuse transmission is drawn FIRST, against the front lobes
	// as a whole. The probability comes from the shared brdf_inc helper -- the
	// same value combinedBRDFPdf folds into both MIS sides, so sampler and pdf
	// cannot drift (EMISSIVE_LIGHTS.md par.4 invariant 2 extended).
	float dt_select_p = diffuseTransmissionSelectProbability(brdf_mat);
	if (brdf_mat.transmissivness > 0.0) {
		// Transmissive surface: the two-way split becomes three-way. For a full
		// delta dielectric the Fresnel reflect/refract choice lives INSIDE the
		// lobe, so there is no selection probability to divide out here.
		// Partial transmission mixing diffuse/specular is not handled yet.
		brdfType = TRANSMISSION_TYPE;
	} else if (dt_select_p > 0.0 && rand(ps.rng_state) < dt_select_p) {
		brdfType = DIFFUSE_TRANSMISSION_TYPE;
		ps.throughput /= dt_select_p;
	} else {
		if (dt_select_p > 0.0) {
			ps.throughput /= (1.0 - dt_select_p);
		}
		if (specular_select_p >= 1.0) {
			brdfType = SPECULAR_TYPE;
		} else if (specular_select_p <= 0.0) {
			brdfType = DIFFUSE_TYPE;
		} else {
			if (rand(ps.rng_state) < specular_select_p) {
				brdfType = SPECULAR_TYPE;
				ps.throughput /= specular_select_p;
			} else {
				brdfType = DIFFUSE_TYPE;
				ps.throughput /= (1.0 - specular_select_p);
			}
		}
	}

	// Camera-vertex BSDF pair comes from the R2 sequence (dim pair 0);
	// bounce vertices and spp>2 extras stay white. The white draw is NOT
	// consumed when LD substitutes — stream position is not semantic.
	vec2 u = (ld_n != LD_N_INVALID)
			? ld_first_vertex(uvec2(gl_LaunchIDEXT.xy), ld_n, 0u)
			: rand2(ps.rng_state);
	vec3 next_dir;
	vec3 brdf_weight;
	if (!evalIndirectCombinedBRDF(u, N, h.geometry_normal, V, brdf_mat, brdfType, next_dir, brdf_weight, vec4(0.0))) {
		// Two failure modes:
		//   1) Sample weight is zero (no contribution). Terminate.
		//   2) Sampled direction is below the geometry plane. Recover by
		//      mirroring across the geometry plane (see brdf_inc.glsl). The
		//      mirror is gated by RT_BELOW_HEMISPHERE_RECOVERY_ENABLED; when
		//      disabled, recovery returns false and we terminate the path.
		vec3 recovered_dir;
		// Never mirror-recover a transmission sample (either lobe): mirroring a
		// through-the-surface direction back above the plane is exactly the
		// "glass becomes a mirror" failure. Terminate instead. This can only
		// happen on a zero-weight sample; the lobes never fail the plane check.
		if (brdfType == TRANSMISSION_TYPE || brdfType == DIFFUSE_TRANSMISSION_TYPE ||
				luminance(brdf_weight) == 0.0 ||
				!recoverBelowHemisphereSample(next_dir, h.geometry_normal, recovered_dir)) {
			ps.packed_bounces_flags = set_path_terminated(ps.packed_bounces_flags);
			path_pack(payload, ps);
			return;
		}
		next_dir = recovered_dir;
	}

	ps.throughput *= brdf_weight;

	if (brdfType == DIFFUSE_TYPE || brdfType == DIFFUSE_TRANSMISSION_TYPE) {
		// Diffuse transmission is a diffuse-lobe sibling: same variance
		// character, same reason to be depth-limited.
		ps.packed_bounces_flags = inc_diffuse_bounce(ps.packed_bounces_flags);
	} else {
		ps.packed_bounces_flags = inc_total_bounce(ps.packed_bounces_flags);
	}

	// MIS pdf transport (HANDOFF_DIFFUSE_EMISSIVES Step 0): carry this
	// bounce's BSDF sampling pdf to the NEXT closest-hit, where the emissive
	// add needs it for the balance heuristic. The fork shades in closest-hit
	// (RTXPT/Falcor shade in raygen and keep PathState in registers), so the
	// payload's free bits are the only channel between consecutive hits —
	// see mis_pdf_store in raytracing_inc.glsl and PAYLOAD_BUDGET.md §3.
	// Delta transmission has no finite pdf: clear the field, and the next
	// emissive hit adds unweighted (weight-1 convention for delta bounces).
	// A mirror-recovered direction reuses the same formula: the recovery is
	// itself an approximation and the pdf follows the direction actually
	// traced.
	if (brdfType == TRANSMISSION_TYPE) {
		ps.packed_bounces_flags = mis_pdf_clear(ps.packed_bounces_flags);
	} else {
		ps.packed_bounces_flags = mis_pdf_store(ps.packed_bounces_flags,
				combinedBRDFPdf(N, V, next_dir, brdf_mat, specular_select_p));
	}

	// Hand the next ray back to raygen. PATH_TERMINATED_FLAG stays clear so
	// the raygen loop continues with the reconstructed origin and next_ray_dir.
	ps.hit_t = gl_HitTEXT;
	// Transmitted rays continue on the FAR side of the surface -- the
	// origin offset must push below the geometry plane, or the continuation
	// ray self-intersects the interface it just refracted through.
	ps.offset_normal = (dot(next_dir, h.geometry_normal) >= 0.0)
			? h.geometry_normal
			: -h.geometry_normal;
	ps.next_ray_dir = next_dir;
	path_pack(payload, ps);
}
