#include "voxel_shape_forge.h"

#include "../../util/godot/core/class_db.h"

namespace zylann::voxel {

namespace {

// ---- primitives: each mirrors its stone_forge.gd namesake exactly ----

inline float sd_box(const Vector3 &p, const Vector3 &half, float rounding) {
	const Vector3 q = p.abs() - (half - Vector3(1, 1, 1) * rounding);
	const Vector3 outside(MAX(q.x, 0.f), MAX(q.y, 0.f), MAX(q.z, 0.f));
	return outside.length() + MIN(MAX(q.x, MAX(q.y, q.z)), 0.f) - rounding;
}

inline float sd_cyl(const Vector3 &p, float r, float half_h, int axis) {
	Vector2 radial;
	float along;
	switch (axis) {
		case 0:
			radial = Vector2(p.y, p.z);
			along = p.x;
			break;
		case 2:
			radial = Vector2(p.x, p.y);
			along = p.z;
			break;
		default:
			radial = Vector2(p.x, p.z);
			along = p.y;
			break;
	}
	const Vector2 d(radial.length() - r, Math::abs(along) - half_h);
	return MIN(MAX(d.x, d.y), 0.f) + Vector2(MAX(d.x, 0.f), MAX(d.y, 0.f)).length();
}

inline float sd_gable(const Vector3 &p, const Vector3 &size) {
	const float hw = size.z * 0.5f;
	const float h = size.y;
	const Vector2 q(Math::abs(p.z), p.y + h * 0.5f);
	const float slope = (q.x * h + q.y * hw - hw * h) / MAX(Math::sqrt(h * h + hw * hw), 0.0001f);
	const float d2 = MAX(slope, -q.y);
	const float dx = Math::abs(p.x) - size.x * 0.5f;
	return MIN(MAX(d2, dx), 0.f) + Vector2(MAX(d2, 0.f), MAX(dx, 0.f)).length();
}

inline float sd_arch(const Vector3 &p, float w, float h, float d) {
	const float r = w * 0.5f;
	const float shoulder = MAX(h - r, 0.001f);
	const float body = sd_box(p - Vector3(0.f, (shoulder - h) * 0.5f, 0.f), Vector3(r, shoulder * 0.5f, d * 0.5f), 0.f);
	const float head = sd_cyl(p - Vector3(0.f, h * 0.5f - r, 0.f), r, d * 0.5f, 2);
	return MIN(body, head);
}

inline float sd_prism(const Vector3 &p, const std::vector<Vector2> &poly, float half_h) {
	const int n = int(poly.size());
	if (n < 3) {
		return 1.0e9f;
	}
	const Vector2 q(p.x, p.z);
	float d = (q - poly[0]).length_squared();
	float sgn = 1.f;
	int j = n - 1;
	for (int i = 0; i < n; ++i) {
		const Vector2 vi = poly[i];
		const Vector2 vj = poly[j];
		const Vector2 e = vj - vi;
		const Vector2 wv = q - vi;
		const Vector2 b = wv - e * CLAMP(wv.dot(e) / MAX(e.length_squared(), 0.000001f), 0.f, 1.f);
		d = MIN(d, b.length_squared());
		const bool c1 = q.y >= vi.y;
		const bool c2 = q.y < vj.y;
		const bool c3 = e.x * wv.y > e.y * wv.x;
		if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) {
			sgn = -sgn;
		}
		j = i;
	}
	const float d2 = sgn * Math::sqrt(d);
	const float dy = Math::abs(p.y) - half_h;
	return MIN(MAX(d2, dy), 0.f) + Vector2(MAX(d2, 0.f), MAX(dy, 0.f)).length();
}

} // namespace

int VoxelShapeForge::_compile(const Dictionary &shape, std::vector<Node> &out) {
	Node n;
	n.at = shape.get("at", Vector3());
	n.yaw = float(shape.get("yaw", 0.0));
	const String op = String(shape.get("op", "box"));
	// Reserve this node's slot before recursing so child indices stay valid.
	const int idx = int(out.size());
	out.push_back(n);
	if (op == "box") {
		out[idx].op = 0;
		out[idx].size = shape.get("size", Vector3(1, 1, 1));
		out[idx].rounding = float(shape.get("rounding", 0.06));
	} else if (op == "cyl") {
		out[idx].op = 1;
		out[idx].r = float(shape.get("r", 0.5));
		out[idx].h = float(shape.get("h", 1.0));
		const String axis = String(shape.get("axis", "y"));
		out[idx].axis = axis == "x" ? 0 : (axis == "z" ? 2 : 1);
	} else if (op == "gable") {
		out[idx].op = 2;
		out[idx].size = shape.get("size", Vector3(1, 1, 1));
	} else if (op == "arch") {
		out[idx].op = 3;
		out[idx].w = float(shape.get("w", 1.0));
		out[idx].arch_h = float(shape.get("h", 2.0));
		out[idx].d = float(shape.get("d", 1.0));
	} else if (op == "prism") {
		out[idx].op = 4;
		out[idx].h = float(shape.get("h", 1.0));
		const Array poly = shape.get("poly", Array());
		for (int i = 0; i < poly.size(); ++i) {
			out[idx].poly.push_back(Vector2(poly[i]));
		}
	} else if (op == "plane") {
		out[idx].op = 5;
		out[idx].plane_n = Vector3(shape.get("n", Vector3(0, 1, 0))).normalized();
		out[idx].plane_d = float(shape.get("d", 0.0));
	} else if (op == "union" || op == "intersect") {
		out[idx].op = op == "union" ? 6 : 7;
		const Array kids = shape.get("of", Array());
		for (int i = 0; i < kids.size(); ++i) {
			const int c = _compile(kids[i], out);
			out[idx].children.push_back(c);
		}
	} else if (op == "subtract") {
		out[idx].op = 8;
		// {} in the `from` slot compiles to the default unit box, exactly as
		// GDScript's shape.get("from", {}) falls through match's "box" arm.
		const int base = _compile(shape.get("from", Dictionary()), out);
		out[idx].children.push_back(base);
		const Array cuts = shape.get("cut", Array());
		for (int i = 0; i < cuts.size(); ++i) {
			const int c = _compile(cuts[i], out);
			out[idx].children.push_back(c);
		}
	} else {
		out[idx].op = -1; // unknown: +1e9, same as the GDScript fallthrough
	}
	return idx;
}

float VoxelShapeForge::_eval(const std::vector<Node> &nodes, const int idx, const Vector3 p) const {
	const Node &n = nodes[idx];
	Vector3 q = p - n.at;
	if (n.yaw != 0.f) {
		q = q.rotated(Vector3(0, 1, 0), -n.yaw);
	}
	switch (n.op) {
		case 0:
			return sd_box(q, n.size * 0.5f, n.rounding);
		case 1:
			return sd_cyl(q, n.r, n.h * 0.5f, n.axis);
		case 2:
			return sd_gable(q, n.size);
		case 3:
			return sd_arch(q, n.w, n.arch_h, n.d);
		case 4:
			return sd_prism(q, n.poly, n.h * 0.5f);
		case 5:
			return q.dot(n.plane_n) - n.plane_d;
		case 6: {
			float u = 1.0e9f;
			for (const int c : n.children) {
				u = MIN(u, _eval(nodes, c, q));
			}
			return u;
		}
		case 7: {
			float i = -1.0e9f;
			for (const int c : n.children) {
				i = MAX(i, _eval(nodes, c, q));
			}
			return i;
		}
		case 8: {
			float base = n.children.empty() ? 1.0e9f : _eval(nodes, n.children[0], q);
			for (size_t k = 1; k < n.children.size(); ++k) {
				base = MAX(base, -_eval(nodes, n.children[k], q));
			}
			return base;
		}
		default:
			return 1.0e9f;
	}
}

VoxelShapeForge::Erosion VoxelShapeForge::_parse_erosion(const Dictionary &erosion, const Ref<Noise> &noise) {
	Erosion e;
	e.strength = float(erosion.get("strength_m", 0.0));
	e.enabled = e.strength > 0.f && noise.is_valid();
	if (!e.enabled) {
		return e;
	}
	e.top_bias = CLAMP(float(erosion.get("top_bias", 0.65)), 0.f, 1.f);
	e.corner_boost = float(erosion.get("corner_boost", 0.8));
	e.keep_bottom = float(erosion.get("keep_bottom_m", 0.25));
	e.base_y = float(erosion.get("base_y", 0.0));
	e.height = MAX(float(erosion.get("height", 1.0)), 0.001f);
	e.centre = Vector3(erosion.get("centre", Vector3()));
	e.half = Vector3(erosion.get("half", Vector3(1, 1, 1)));
	return e;
}

float VoxelShapeForge::_eroded(
		const std::vector<Node> &nodes, const Erosion &e, const Ref<Noise> &noise, const Vector3 p) const {
	const float d = _eval(nodes, 0, p);
	if (!e.enabled) {
		return d;
	}
	// Same exposure model as the GDScript closure: weather falls on what the
	// sky sees, the footing is spared, vertical corners shed first.
	const float up01 = CLAMP((p.y - e.base_y) / e.height, 0.f, 1.f);
	float exposure = Math::lerp(1.f - e.top_bias, 1.f, up01);
	exposure *= float(Math::smoothstep(0.f, MAX(e.keep_bottom, 0.001f), float(p.y - e.base_y)));
	const float cx = CLAMP(Math::abs(p.x - e.centre.x) / MAX(e.half.x, 0.001f), 0.f, 1.f);
	const float cz = CLAMP(Math::abs(p.z - e.centre.z) / MAX(e.half.z, 0.001f), 0.f, 1.f);
	exposure *= 1.f + e.corner_boost * cx * cz;
	const float nv = noise->get_noise_3d(p.x, p.y, p.z) * 0.5f + 0.5f;
	return d + MAX(0.f, nv - 0.35f) / 0.65f * e.strength * exposure;
}

void VoxelShapeForge::fill_sdf(Ref<godot::VoxelBuffer> buffer, Dictionary shape, Vector3 sample_min_m,
		float voxel_m, Dictionary erosion, Ref<Noise> noise) {
	ERR_FAIL_COND(buffer.is_null());
	ERR_FAIL_COND(voxel_m <= 0.f);
	std::vector<Node> nodes;
	_compile(shape, nodes);
	const Erosion e = _parse_erosion(erosion, noise);
	const Vector3i dims = buffer->get_size();
	const float inv_voxel = 1.f / voxel_m;
	for (int z = 0; z < dims.z; ++z) {
		for (int y = 0; y < dims.y; ++y) {
			for (int x = 0; x < dims.x; ++x) {
				const Vector3 p = sample_min_m + Vector3(x, y, z) * voxel_m;
				buffer->set_voxel_f(_eroded(nodes, e, noise, p) * inv_voxel, x, y, z, godot::VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
}

float VoxelShapeForge::sample(Dictionary shape, Vector3 p, Dictionary erosion, Ref<Noise> noise) {
	std::vector<Node> nodes;
	_compile(shape, nodes);
	return _eroded(nodes, _parse_erosion(erosion, noise), noise, p);
}

void VoxelShapeForge::_bind_methods() {
	ClassDB::bind_method(
			D_METHOD("fill_sdf", "buffer", "shape", "sample_min_m", "voxel_m", "erosion", "noise"),
			&VoxelShapeForge::fill_sdf);
	ClassDB::bind_method(D_METHOD("sample", "shape", "p", "erosion", "noise"), &VoxelShapeForge::sample);
}

} // namespace zylann::voxel
