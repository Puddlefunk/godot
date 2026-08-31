#ifndef VOXEL_SHAPE_FORGE_H
#define VOXEL_SHAPE_FORGE_H

#include "../../storage/voxel_buffer_gd.h"
#include "../../util/godot/classes/noise.h"

#include <vector>

namespace zylann::voxel {

// Native sampler for terrain-scratch's StoneForge shape trees (stone_forge.gd).
//
// StoneForge describes parts as Dictionary shape trees -- box / cyl / gable /
// arch / prism / plane primitives with optional per-node `at` and `yaw`,
// composed by union / intersect / subtract -- plus an erosion envelope that
// pushes the surface inward where it is most exposed. The GDScript evaluator
// (`shape_sdf` + the closure in `build_shape_mesh`) is the SEMANTIC REFERENCE
// and stays; this class is a bit-faithful port of it that fills a whole
// VoxelBuffer in one call, replacing the per-voxel Callable loop that was the
// entire cost of a bake (~1M samples/s in GDScript).
//
// Parity rules, in order of importance:
// * Every formula mirrors stone_forge.gd exactly, including defaults ({} in a
//   `from` slot is a unit box, unknown ops are +1e9, `intersect` starts at
//   -1e9). If the GDScript changes, this must change with it -- the script
//   asserts parity on random points in debug builds, so drift is loud.
// * The erosion noise is the CALLER'S FastNoiseLite (passed as Ref<Noise>),
//   never one built here, so native and script sample the identical stream.
// * SDF values are written to CHANNEL_SDF in VOXEL units (divided by
//   voxel_m), matching the convention _mesh_sdf established.
//
// Instance methods rather than statics so stone_forge.gd can reach the class
// through ClassDB.instantiate() and keep parsing on engine builds that
// predate it.
class VoxelShapeForge : public RefCounted {
	GDCLASS(VoxelShapeForge, RefCounted)

public:
	// Fills the whole buffer's SDF channel from the shape tree.
	// `sample_min_m` is the world/local position of voxel (0,0,0) -- the
	// caller's `sample_min_m`, padding already included.
	// `erosion` carries the envelope: strength_m, top_bias, corner_boost,
	// keep_bottom_m, base_y, height, centre (Vector3), half (Vector3).
	// Empty dictionary or null noise = raw SDF, no weathering.
	void fill_sdf(Ref<godot::VoxelBuffer> buffer, Dictionary shape, Vector3 sample_min_m, float voxel_m,
			Dictionary erosion, Ref<Noise> noise);

	// One point, same evaluation -- exists for the debug parity assert.
	float sample(Dictionary shape, Vector3 p, Dictionary erosion, Ref<Noise> noise);

protected:
	static void _bind_methods();

private:
	struct Node {
		int op = 0; // 0 box, 1 cyl, 2 gable, 3 arch, 4 prism, 5 plane, 6 union, 7 intersect, 8 subtract
		Vector3 at;
		float yaw = 0.f;
		Vector3 size = Vector3(1, 1, 1); // box, gable
		float rounding = 0.06f; // box
		float r = 0.5f, h = 1.f; // cyl (h also prism height)
		int axis = 1; // cyl: 0 x, 1 y, 2 z
		float w = 1.f, arch_h = 2.f, d = 1.f; // arch
		Vector3 plane_n = Vector3(0, 1, 0);
		float plane_d = 0.f;
		std::vector<Vector2> poly; // prism
		std::vector<int> children; // union/intersect: of[]; subtract: [from, cut...]
	};

	struct Erosion {
		bool enabled = false;
		float strength = 0.f, top_bias = 0.65f, corner_boost = 0.8f, keep_bottom = 0.25f;
		float base_y = 0.f, height = 1.f;
		Vector3 centre, half;
	};

	int _compile(const Dictionary &shape, std::vector<Node> &out);
	float _eval(const std::vector<Node> &nodes, int idx, Vector3 p) const;
	static Erosion _parse_erosion(const Dictionary &erosion, const Ref<Noise> &noise);
	float _eroded(const std::vector<Node> &nodes, const Erosion &e, const Ref<Noise> &noise, Vector3 p) const;
};

} // namespace zylann::voxel

#endif // VOXEL_SHAPE_FORGE_H
