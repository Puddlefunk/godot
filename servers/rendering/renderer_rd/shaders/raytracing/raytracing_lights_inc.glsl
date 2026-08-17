// Light sampling and Next Event Estimation (NEE) for raytracing.
// Requires: raytracing_inc.glsl, brdf_inc.glsl, tlas at binding 1, payload at location 0.
// Note: ray_query_alpha_test() still requires GL_EXT_ray_query (used by DLSS-RR path or when USE_RAY_QUERY_SHADOWS is defined).

// ============================================================================
// Light Types and Constants
// ============================================================================

#define RT_LIGHT_TYPE_OMNI 0 // Point light with radius (soft shadows)
#define RT_LIGHT_TYPE_DIRECTIONAL 1 // Sun/moon with angular size
#define RT_LIGHT_TYPE_SPOT 3 // Spot light with cone falloff

// Reservoir sampling batch size for stochastic light selection.
#ifndef RT_LIGHT_RESERVOIR_SIZE
#define RT_LIGHT_RESERVOIR_SIZE 16
#endif

// ============================================================================
// Light Data (matches C++ RT_LightData, 80 bytes, std430)
// ============================================================================

struct RTLightData {
	vec3 position; // World pos (omni/spot) or direction (directional).
	uint type; // RT_LIGHT_TYPE_*.
	vec3 emission; // HDR emission color (color * energy).
	float radius; // Light size (omni/spot) or angular radius (directional).
	float attenuation; // Attenuation exponent (2=inverse-square, 1=linear).
	float inv_max_range; // 1/range (-1 = infinite).
	float max_range_squared; // range^2 (0 = infinite).
	float specular_amount; // Godot specular multiplier [0..1].
	float indirect_energy; // Godot indirect energy multiplier.
	float inv_spot_attenuation; // Spot cone softness.
	float cos_spot_angle; // Cosine of spot cone half-angle.
	float _pad0;
	vec3 spot_direction; // Spot direction (normalized, world space).
	float _pad1;
};

// Light buffer SSBO (binding provided by the including shader via RT_LIGHT_BUFFER_BINDING).
#ifndef RT_LIGHT_BUFFER_BINDING
#define RT_LIGHT_BUFFER_BINDING 13
#endif

layout(set = 0, binding = RT_LIGHT_BUFFER_BINDING, std430) readonly buffer LightBuffer {
	RTLightData rt_lights[];
};

// ============================================================================
// Unified Cone Sampling (for sphere, directional, spot lights)
// ============================================================================

struct LightSample {
	vec3 cone_axis; // Direction to sample around (normalized).
	float cos_theta_max; // Cone half-angle cosine (1.0 = point, 0.0 = hemisphere).
	vec3 emission; // Light radiance.
	float distance_sq; // Squared distance to light (0 = directional).
	float max_distance; // Max shadow ray distance.
};

// Build orthonormal basis from a single direction.
void lights_build_basis(vec3 dir, out vec3 tangent, out vec3 bitangent) {
	vec3 up = abs(dir.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
	tangent = normalize(cross(up, dir));
	bitangent = cross(dir, tangent);
}

// Transform local direction to world space around axis.
vec3 lights_local_to_world(vec3 local_dir, vec3 axis) {
	vec3 tangent, bitangent;
	lights_build_basis(axis, tangent, bitangent);
	return local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * axis;
}

// Prepare unified cone sample from any light type.
LightSample lights_prepare_sample(vec3 hit_pos, RTLightData light) {
	LightSample s;

	vec3 to_light = light.position - hit_pos;
	float dist_sq = dot(to_light, to_light);
	float inv_dist = inversesqrt(dist_sq + 1e-10);
	float dist = dist_sq * inv_dist;

	float is_directional = (light.type == RT_LIGHT_TYPE_DIRECTIONAL) ? 1.0 : 0.0;

	// Cone axis.
	vec3 sphere_axis = to_light * inv_dist;
	vec3 dir_axis = -normalize(light.position);
	s.cone_axis = mix(sphere_axis, dir_axis, is_directional);

	// Distance (0 for directional = no falloff).
	s.distance_sq = mix(dist_sq, 0.0, is_directional);

	// Cone angle from subtended solid angle.
	float sin_theta_sphere = clamp(light.radius * inv_dist, 0.0, 1.0);
	float cos_theta_sphere = sqrt(max(0.0, 1.0 - sin_theta_sphere * sin_theta_sphere));
	float cos_theta_dir = cos(light.radius);
	s.cos_theta_max = mix(cos_theta_sphere, cos_theta_dir, is_directional);
	s.cos_theta_max = min(s.cos_theta_max, 0.999999);

	s.emission = light.emission;

	// Max shadow ray distance.
	float sphere_max = dist + light.radius;
	float dir_max = 10000.0;
	s.max_distance = mix(sphere_max, dir_max, is_directional);

	return s;
}

// Sample a direction within the light's cone.
vec3 lights_sample_cone(LightSample ls, vec2 u, out float pdf) {
	float cos_theta = 1.0 - u.x * (1.0 - ls.cos_theta_max);
	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
	float phi = 2.0 * PI * u.y;

	vec3 local_dir = vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
	vec3 L = lights_local_to_world(local_dir, ls.cone_axis);

	float solid_angle = 2.0 * PI * (1.0 - ls.cos_theta_max);
	pdf = 1.0 / max(solid_angle, 1e-10);

	return L;
}

// ============================================================================
// Attenuation
// ============================================================================

// Godot-style windowed distance attenuation.
// window = (1 - (d/range)^4)^2, combined with pow(d, -decay).
float lights_get_attenuation(LightSample ls, float inv_max_range, float decay) {
	if (ls.distance_sq <= 0.0) {
		return 1.0; // Directional: no distance attenuation.
	}

	float distance = sqrt(ls.distance_sq);
	float atten = min(pow(max(distance, 0.0001), -decay), 1.0);

	// Windowed falloff if range is finite (inv_max_range >= 0).
	if (inv_max_range >= 0.0) {
		float nd = distance * inv_max_range;
		nd *= nd;
		nd *= nd; // nd^4
		nd = max(1.0 - nd, 0.0);
		nd *= nd; // nd^2 window
		return atten * nd;
	}

	return atten;
}

// Per-light specular multiplier.
float lights_get_specular_multiplier(float specular_amount, float roughness) {
	if (specular_amount >= 0.0) {
		return specular_amount;
	} else {
		float r3 = roughness * roughness * roughness;
		return mix(0.0, r3, -specular_amount);
	}
}

// ============================================================================
// Inline Alpha Test (shared by all ray query proceed loops)
// ============================================================================

/// Inline alpha test for ray query candidates. Returns true if the hit is opaque (alpha >= 0.5).
/// Mirrors the any-hit shader logic for use with inline ray queries.
bool ray_query_alpha_test(uint geometry_idx, uint primitive_id, vec2 candidate_bary) {
	vec3 bary = vec3(1.0 - candidate_bary.x - candidate_bary.y, candidate_bary.x, candidate_bary.y);

	GeometryData geom = geometries[geometry_idx];
	uint i0, i1, i2;
	get_triangle_indices_ex(geom, primitive_id, i0, i1, i2);
	vec2 uv = fetch_uv(geom, i0, i1, i2, bary);

	MaterialData mat = materials[geometry_idx];
	uv = uv * mat.uv1_scale + mat.uv1_offset;
	float alpha = texture(sampler2D(bindless_textures[nonuniformEXT(mat.albedo_texture_idx)], SAMPLER_LINEAR_WITH_MIPMAPS_REPEAT), uv).a;
	alpha *= mat.albedo_color.a;

	return alpha >= 0.5;
}

// ============================================================================
// Shadow Ray (traceRayEXT pipeline)
// ============================================================================

/// Returns per-channel light transmittance along the shadow ray:
/// vec3(1) = fully visible, vec3(0) = occluded by opaque geometry, and in
/// between = Beer-Lambert attenuation through transmissive contract media.
/// Shadow rays stay STRAIGHT -- no refraction, no caustics; the standard PT
/// shadow contract.
///
/// Uses SkipClosestHitShader so only any_hit and miss are invoked.
/// TerminateOnFirstHit still early-exits on the first confirmed OPAQUE hit;
/// transmissive hits are ignored by the any-hit (traversal continues) after
/// accumulating signed optical depth into the payload's three non-radiance
/// 32-bit slots (hit_t = R; the two oct slots reinterpreted as raw fp32 = G, B).
/// Signed: +sigma*t at back faces, -sigma*t at front faces, so the sum equals
/// sigma * total interior length regardless of the (unordered) any-hit
/// invocation order. fp32 on purpose -- fp16 payload channels clamp and lose
/// the log-domain range.
///
/// USE_RAY_QUERY_SHADOWS returns binary visibility only. Inline ray queries
/// never invoke the any-hit shader, so there is no hook to accumulate optical
/// depth; transmissive media cast fully opaque shadows on that path, alongside
/// its existing inability to handle complex alpha materials.
vec3 lights_trace_shadow_ray(vec3 origin, vec3 direction, float max_dist, inout uint rng_state) {
#ifdef USE_RAY_QUERY_SHADOWS
	// Ray queries are significantly faster, but can not handle complex alpha materials
	rayQueryEXT shadow_rq;
	rayQueryInitializeEXT(shadow_rq, tlas,
			gl_RayFlagsTerminateOnFirstHitEXT,
			RT_MASK_SHADOW, origin, 0.001, direction, max_dist - 0.001);

	while (rayQueryProceedEXT(shadow_rq)) {
		if (rayQueryGetIntersectionTypeEXT(shadow_rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
			// quick and dirty way to check transparency by sampling the alpha texture
			// this completely ignores the actual material, so might not be accurate
			if (ray_query_alpha_test(
						rayQueryGetIntersectionInstanceCustomIndexEXT(shadow_rq, false),
						rayQueryGetIntersectionPrimitiveIndexEXT(shadow_rq, false),
						rayQueryGetIntersectionBarycentricsEXT(shadow_rq, false))) {
				rayQueryConfirmIntersectionEXT(shadow_rq);
			}
		}
	}

	return (rayQueryGetIntersectionTypeEXT(shadow_rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
			? vec3(1.0)
			: vec3(0.0);
#else
	// Save full payload, set up shadow ray, then restore after trace.
	// (The SER path now also uses the payload for the accumulator and the
	// any-hit's is_shadow_ray check, so both paths save/init/restore.)
	PathPayload saved_payload = payload;

	PathState shadow_ps;
	shadow_ps.radiance = vec3(0.0);
	shadow_ps.throughput = vec3(0.0);
	shadow_ps.packed_bounces_flags = set_shadow_ray(0u);
	shadow_ps.rng_state = 0u;
	shadow_ps.hit_t = 0.0;
	shadow_ps.offset_normal = vec3(0.0, 0.0, 1.0);
	shadow_ps.next_ray_dir = vec3(0.0, 0.0, 1.0);
	path_pack(payload, shadow_ps);
	// Zero the accumulator AFTER path_pack -- it just seeded the oct slots with
	// encoded normals, and those bits must start as fp32 zero.
	payload.hit_t = 0.0;
	payload.oct_offset_nrm = floatBitsToUint(0.0);
	payload.oct_next_dir = floatBitsToUint(0.0);

#ifdef USE_SER
	hitObjectEXT hitObject;
	hitObjectTraceRayEXT(hitObject, tlas,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
			RT_MASK_SHADOW, 0, 0, 0,
			origin, 0.001, direction, max_dist - 0.001, 0);

	bool visible = !(hitObjectIsHitEXT(hitObject));
#else
	/// The miss shader writes radiance.x = 1.0 for shadow rays (visible).
	/// If an opaque hit occurs, miss is never called and radiance stays 0.
	// Shadow rays only test SHADOW-participating instances: cast_shadow Off
	// drops the bit, Shadows Only carries ONLY this bit.
	traceRayEXT(tlas,
			gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
			RT_MASK_SHADOW, 0, 0, 0,
			origin, 0.001, direction, max_dist - 0.001, 0);

	bool visible = unpackHalf2x16(payload.packed_rt[0]).x > 0.5;
#endif

	vec3 optical_depth = vec3(
			payload.hit_t,
			uintBitsToFloat(payload.oct_offset_nrm),
			uintBitsToFloat(payload.oct_next_dir));

	payload = saved_payload;

	if (!visible) {
		return vec3(0.0);
	}
	// Negative net depth appears when the ray STARTS inside a medium (its entry
	// face is behind the origin, so the pair is unbalanced) -- clamp to the
	// physical range instead of amplifying.
	return exp(-max(optical_depth, vec3(0.0)));
#endif
}

// ============================================================================
// Emissive Triangle Lights (HANDOFF_DIFFUSE_EMISSIVES Steps 1-3)
// Host-assembled per-frame table (binding 29) + reverse map (binding 30).
// ============================================================================

// Matches C++ RT_EmissiveTriData (96 B, std430, scalars only).
struct RTEmissiveTriData {
	float pos0_x, pos0_y, pos0_z; // world space
	float pos1_x, pos1_y, pos1_z;
	float pos2_x, pos2_y, pos2_z;
	uint color_packed_0, color_packed_1, color_packed_2; // rgba8 per vertex (unpackUnorm4x8)
	uint uv_packed_0, uv_packed_1, uv_packed_2; // 2xfp16 per vertex (unpackHalf2x16)
	uint geometry_idx; // into geometries[]/materials[], THIS frame
	uint prim_id;
	float pulse_phase;
	float area_ws;
	// pdf INVARIANT: this is the probability the alias table actually uses.
	// The sampler below divides by it; the reverse lookup reports it. A
	// stale table costs efficiency; a mismatch between the two costs
	// correctness.
	float selection_pdf;
	float alias_prob; // Vose: keep this entry if u < alias_prob, else alias_idx
	uint alias_idx;
	float _pad0;
	float _pad1;
};

layout(set = 0, binding = 29, std430) readonly buffer EmissiveTriBuffer {
	RTEmissiveTriData emissive_tris[];
};

// Reverse map: header of {segment offset, prim count} pairs per geometry_idx,
// then per-geometry segments mapping prim_id -> table entry (or NONE).
layout(set = 0, binding = 30, std430) readonly buffer EmissiveMapBuffer {
	uint emissive_map[];
};

#define RT_EMISSIVE_NONE 0xFFFFFFFFu

/// Reverse lookup for MIS: the FULL solid-angle pdf (pool probability
/// included) with which the NEE sampler below would generate the hit
/// described by the arguments. 0.0 = the light sampler cannot generate this
/// hit (not in the table, wrong-facing, table empty) => the balance weight at
/// the emissive add is exactly 1.0, i.e. hit-only behaviour for that path.
float emissive_triangle_pdf(uint geometry_idx, uint prim_id, vec3 ray_origin, vec3 ray_dir, float hit_t, vec3 hit_geometry_normal) {
	uint tri_count = uint(get_rt_param(RT_PARAM_EMISSIVE_TRI_COUNT));
	if (tri_count == 0u) {
		return 0.0;
	}
	float pool_prob = get_rt_param(RT_PARAM_EMISSIVE_POOL_PROB);
	if (pool_prob <= 0.0) {
		return 0.0;
	}
	uint off = emissive_map[geometry_idx * 2u];
	if (off == RT_EMISSIVE_NONE) {
		return 0.0;
	}
	uint cnt = emissive_map[geometry_idx * 2u + 1u];
	if (prim_id >= cnt) {
		return 0.0;
	}
	uint entry_idx = emissive_map[off + prim_id];
	if (entry_idx == RT_EMISSIVE_NONE) {
		return 0.0;
	}
	RTEmissiveTriData e = emissive_tris[entry_idx];
	// Two-sided emitters (|cos|): matches the hit path, which adds emission
	// regardless of facing; wrong-side NEE samples shadow out against the
	// emitter's own body, so the conventions agree.
	float cos_l = abs(dot(hit_geometry_normal, ray_dir));
	if (cos_l < 1e-4 || e.area_ws <= 1e-12) {
		return 0.0;
	}
	// Area-measure pdf (selection / area) converted to solid angle.
	return pool_prob * e.selection_pdf * (hit_t * hit_t) / (e.area_ws * cos_l);
}

// ============================================================================
// Next Event Estimation (NEE) - Direct Light Sampling
// ============================================================================

/// One NEE sample from the emissive triangle pool: alias-table selection,
/// uniform point on the triangle, canonical glow evaluated at the sampled
/// barycentric (the rim→cap→tip ramp survives sampling exactly), per-channel
/// shadow transmittance, MIS balance weight against BSDF sampling.
/// Returns the fully weighted, pdf-divided contribution.
/// hit_pos is the RAW hit position; the shadow-ray origin is offset per
/// sample along ±geo_n depending on which side of the surface L leaves
/// through (Step 2b: back-side samples feed the diffuse-transmission lobe).
vec3 lights_sample_emissive_triangle(
		vec3 hit_pos,
		vec3 geo_n,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		inout uint rng_state,
		float specular_select_p,
		uint tri_count,
		float pool_prob,
		uint ld_n) {
	// O(1) alias-table selection.
	uint i = min(uint(rand(rng_state) * float(tri_count)), tri_count - 1u);
	RTEmissiveTriData e = emissive_tris[i];
	if (rand(rng_state) >= e.alias_prob) {
		e = emissive_tris[e.alias_idx];
	}
	if (e.selection_pdf <= 0.0 || e.area_ws <= 1e-12) {
		return vec3(0.0);
	}

	// Uniform point on the triangle (sqrt warp). Camera-vertex draws use the
	// R2 sequence (dim pair 1 — Step 4 inc. B); alias-table selection above
	// stays white (stratifying a discrete jump table scrambles rather than
	// stratifies).
	vec3 p0 = vec3(e.pos0_x, e.pos0_y, e.pos0_z);
	vec3 p1 = vec3(e.pos1_x, e.pos1_y, e.pos1_z);
	vec3 p2 = vec3(e.pos2_x, e.pos2_y, e.pos2_z);
	vec2 u = (ld_n != LD_N_INVALID)
			? ld_first_vertex(uvec2(gl_LaunchIDEXT.xy), ld_n, 1u)
			: rand2(rng_state);
	float su = sqrt(u.x);
	vec3 bary = vec3(1.0 - su, su * (1.0 - u.y), su * u.y);
	vec3 P = p0 * bary.x + p1 * bary.y + p2 * bary.z;

	vec3 to_p = P - hit_pos;
	float dist_sq = dot(to_p, to_p);
	if (dist_sq < 1e-6) {
		return vec3(0.0); // Degenerate: sample point at the receiver.
	}
	float dist = sqrt(dist_sq);
	vec3 L = to_p / dist;

	// Back-side samples are rejected UNLESS the material carries the
	// thin-walled diffuse-transmission lobe -- then they are exactly the
	// membrane-lit-from-behind case this sampler exists for (Step 2b), and
	// evalCombinedBRDFSeparate below returns the BTDF for them.
	if (dot(N, L) <= 0.0 && luminance(material.diffuseTransmission) <= 0.0) {
		return vec3(0.0);
	}

	// Two-sided emitter (|cos|), matching the hit path's facing-agnostic
	// emissive add; wrong-side samples shadow out against the emitter body.
	vec3 tri_n = normalize(cross(p1 - p0, p2 - p0));
	float cos_l = abs(dot(tri_n, L));
	if (cos_l < 1e-4) {
		return vec3(0.0);
	}

	// FULL solid-angle pdf — composition IDENTICAL to emissive_triangle_pdf
	// above (the pdf invariant lives or dies on these two staying twins).
	float p_light = pool_prob * e.selection_pdf * dist_sq / (e.area_ws * cos_l);
	if (!(p_light > 0.0)) {
		return vec3(0.0);
	}

	// Canonical emission at the sampled barycentric.
	vec4 vc = unpackUnorm4x8(e.color_packed_0) * bary.x +
			unpackUnorm4x8(e.color_packed_1) * bary.y +
			unpackUnorm4x8(e.color_packed_2) * bary.z;
	vec2 vuv = unpackHalf2x16(e.uv_packed_0) * bary.x +
			unpackHalf2x16(e.uv_packed_1) * bary.y +
			unpackHalf2x16(e.uv_packed_2) * bary.z;
	vec3 emission = pt_glow_evaluate(materials[e.geometry_idx], vc, vuv, scene_data_block.data.time, e.pulse_phase) *
			scene_data_block.data.emissive_exposure_normalization;
	if (dot(emission, emission) <= 0.0) {
		return vec3(0.0);
	}

	// Per-channel transmittance: mushroom light through a crystal arrives
	// tinted, for free. Origin offset follows the side L exits through: a
	// back-side (transmission) sample offset to the FRONT would immediately
	// re-cross its own triangle and self-occlude. Front samples get the
	// identical origin the pre-2b code used.
	vec3 shadow_origin = offset_ray_origin(hit_pos, (dot(L, geo_n) >= 0.0) ? geo_n : -geo_n);
	vec3 shadow_transmittance = lights_trace_shadow_ray(shadow_origin, L, dist, rng_state);
	if (all(lessThan(shadow_transmittance, vec3(0.0001)))) {
		return vec3(0.0);
	}

	vec3 brdf_diffuse, brdf_specular;
	evalCombinedBRDFSeparate(N, L, V, material, brdf_diffuse, brdf_specular);
	vec3 brdf_value = brdf_diffuse + brdf_specular;

	// MIS balance weight vs BSDF sampling. The BSDF pdf goes through the
	// payload quantizer ROUNDTRIP so this weight and the hit-side weight sum
	// to exactly 1 (raytracing_inc.glsl).
	float p_bsdf = mis_pdf_roundtrip(combinedBRDFPdf(N, V, L, material, specular_select_p));
	float w = p_light / (p_light + p_bsdf);

	return brdf_value * emission * shadow_transmittance * (w / p_light);
}

// Evaluate direct lighting using NEE with stochastic light selection:
// one sample per call, drawn from the analytic pool (mini-batch reservoir,
// as ever) or the emissive triangle pool (alias table), chosen by the
// host-computed power ratio in emissive_pool_prob.
// hit_pos is the RAW hit position; shadow-ray origins are offset per sample
// along ±geo_n by the side L exits through (see lights_sample_emissive_triangle).
vec3 lights_evaluate_direct_lighting(
		vec3 hit_pos,
		vec3 geo_n,
		vec3 N,
		vec3 V,
		MaterialProperties material,
		inout uint rng_state,
		bool is_indirect_bounce,
		uint light_count,
		uint emissive_tri_count,
		float emissive_pool_prob,
		float specular_select_p,
		uint ld_n) {
	bool has_analytic = light_count > 0u;
	bool has_emissive = emissive_tri_count > 0u;
	if (!has_analytic && !has_emissive) {
		return vec3(0.0);
	}

	// Pool selection. The pdf of whichever pool is chosen divides the
	// contribution; analytic lights have no BSDF-sampling partner (they are
	// not geometry), so their MIS weight stays 1 and only the emissive pool
	// carries a balance weight.
	// The host is the SINGLE authority for pool_e (already clamped/edge-cased
	// there); do NOT reshape it here — the reverse lookup above reads the
	// same raw param, and any disagreement between the two is pdf bias.
	float pool_e = emissive_pool_prob;
	if (has_emissive && (pool_e >= 1.0 || rand(rng_state) < pool_e)) {
		// NOTE: p_light inside the helper already includes pool_e — it is
		// the FULL pdf, so no further division here (and the hit-side
		// reverse lookup reports the same full pdf; the twins must agree).
		return lights_sample_emissive_triangle(hit_pos, geo_n, N, V, material, rng_state, specular_select_p, emissive_tri_count, pool_e, ld_n);
	}
	float analytic_inv_prob = 1.0 / (1.0 - pool_e);

	// Mini-batch reservoir sampling: sample k lights, pick the best valid one.
	const uint k = uint(min(RT_LIGHT_RESERVOIR_SIZE, int(light_count)));
	uint valid_found = 0u;
	uint selected_idx = 0u;

	for (uint i = 0u; i < k; i++) {
		uint idx = min(uint(rand(rng_state) * float(light_count)), light_count - 1u);
		RTLightData test_light = rt_lights[idx];

		// Range check for positional lights.
		bool is_positional = (test_light.type == RT_LIGHT_TYPE_OMNI || test_light.type == RT_LIGHT_TYPE_SPOT);
		bool is_valid = !is_positional;
		if (!is_valid) {
			vec3 to_l = test_light.position - hit_pos;
			float d2 = dot(to_l, to_l);
			is_valid = (test_light.max_range_squared == 0.0 || d2 <= test_light.max_range_squared);
		}

		if (is_valid) {
			valid_found++;
			if (rand(rng_state) < 1.0 / float(valid_found)) {
				selected_idx = idx;
			}
		}
	}

	if (valid_found == 0u) {
		return vec3(0.0);
	}

	// Estimate valid count from sample ratio, PDF = 1/validCount.
	float valid_count_estimate = float(light_count) * (float(valid_found) / float(k));
	float light_select_pdf = 1.0 / max(valid_count_estimate, 1.0);

	RTLightData light = rt_lights[selected_idx];
	// Camera-vertex analytic-light sample (cone/sphere) from the R2 sequence
	// (dim pair 2); reservoir selection above stays white — same discrete-
	// draw reasoning as the alias table.
	vec2 u = (ld_n != LD_N_INVALID)
			? ld_first_vertex(uvec2(gl_LaunchIDEXT.xy), ld_n, 2u)
			: rand2(rng_state);

	// === POSITIONAL LIGHT PATH (omni + spot) ===
	if (light.type == RT_LIGHT_TYPE_OMNI || light.type == RT_LIGHT_TYPE_SPOT) {
		vec3 to_light = light.position - hit_pos;
		float dist_sq = dot(to_light, to_light);

		// Early out: outside max range.
		if (light.max_range_squared != 0.0 && dist_sq > light.max_range_squared) {
			return vec3(0.0);
		}

		float dist = sqrt(dist_sq);
		vec3 L;
		float shadow_dist;

		if (light.radius <= 0.01) {
			// True point light: exact direction.
			L = to_light / max(dist, 0.0001);
			shadow_dist = dist;
		} else {
			// Sphere light: cone sampling for soft shadows.
			LightSample ls = lights_prepare_sample(hit_pos, light);
			float light_pdf;
			L = lights_sample_cone(ls, u, light_pdf);
			float t_center = dot(to_light, L);
			vec3 perp = to_light - t_center * L;
			float perp_sq = dot(perp, perp);
			float dt = sqrt(max(0.0, light.radius * light.radius - perp_sq));
			shadow_dist = max(0.0, t_center - dt);
		}

		// Spot cone early-out.
		float spot_atten = 1.0;
		if (light.type == RT_LIGHT_TYPE_SPOT) {
			float scos = dot(-L, light.spot_direction);
			if (scos <= light.cos_spot_angle) {
				return vec3(0.0);
			}
			float spot_rim = max(1e-4, (1.0 - scos) / (1.0 - light.cos_spot_angle));
			spot_atten = 1.0 - pow(spot_rim, light.inv_spot_attenuation);
		}

		// Per-channel transmittance, not a binary visibility test. Origin
		// offset follows the side L exits through (diffuse transmission
		// samples leave through the back).
		vec3 shadow_origin = offset_ray_origin(hit_pos, (dot(L, geo_n) >= 0.0) ? geo_n : -geo_n);
		vec3 shadow_transmittance = lights_trace_shadow_ray(shadow_origin, L, shadow_dist, rng_state);
		if (all(lessThan(shadow_transmittance, vec3(0.0001)))) {
			return vec3(0.0);
		}

		// Evaluate BRDF (diffuse + specular separately for specular_amount control).
		vec3 brdf_diffuse, brdf_specular;
		evalCombinedBRDFSeparate(N, L, V, material, brdf_diffuse, brdf_specular);

		// Distance attenuation.
		LightSample ls_atten;
		ls_atten.distance_sq = dist_sq;
		float atten = lights_get_attenuation(ls_atten, light.inv_max_range, light.attenuation) * spot_atten;

		float spec_mul = lights_get_specular_multiplier(light.specular_amount, material.roughness);
		vec3 brdf_value = brdf_diffuse + brdf_specular * spec_mul;

		float indirect_mul = is_indirect_bounce ? light.indirect_energy : 1.0;

		// NdotL is already included in brdf_value (evalLambertian/evalMicrofacet bake it in).
		vec3 contribution = brdf_value * light.emission * atten * indirect_mul * shadow_transmittance;
		return contribution * analytic_inv_prob / max(light_select_pdf, 1e-10);
	}
	// === CONE LIGHT PATH (directional) ===
	else {
		LightSample ls = lights_prepare_sample(hit_pos, light);
		float light_pdf;
		vec3 L = lights_sample_cone(ls, u, light_pdf);

		float NdotL = dot(N, L);
		// Back-side rejection relaxed for the diffuse-transmission lobe
		// (Step 2b): sun through a membrane is this exact path.
		if (NdotL <= 0.0 && luminance(material.diffuseTransmission) <= 0.0) {
			return vec3(0.0);
		}

		// Per-channel transmittance, not a binary visibility test. Origin
		// offset follows the side L exits through.
		vec3 shadow_origin = offset_ray_origin(hit_pos, (dot(L, geo_n) >= 0.0) ? geo_n : -geo_n);
		vec3 shadow_transmittance = lights_trace_shadow_ray(shadow_origin, L, ls.max_distance, rng_state);
		if (all(lessThan(shadow_transmittance, vec3(0.0001)))) {
			return vec3(0.0);
		}

		vec3 brdf_diffuse, brdf_specular;
		evalCombinedBRDFSeparate(N, L, V, material, brdf_diffuse, brdf_specular);

		float spec_mul = lights_get_specular_multiplier(light.specular_amount, material.roughness);
		vec3 brdf_value = brdf_diffuse + brdf_specular * spec_mul;

		float indirect_mul = is_indirect_bounce ? light.indirect_energy : 1.0;

		return brdf_value * light.emission * indirect_mul * shadow_transmittance * analytic_inv_prob / max(light_select_pdf, 1e-10);
	}
}

