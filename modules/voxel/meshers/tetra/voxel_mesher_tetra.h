#ifndef VOXEL_MESHER_TETRA_H
#define VOXEL_MESHER_TETRA_H

#include "../voxel_mesher.h"

namespace zylann::voxel {

// Marching-tetrahedra mesher with a deterministic "Stålberg" stylization pass.
//
// Ported from the Synthesis project's GDScript implementation (VoxelPlayground.gd).
// Each cube cell of the sample lattice is split into six positively-oriented
// tetrahedra; surface vertices are welded by the unordered pair of lattice-point
// ids they interpolate between, then displaced by a stylization warp of the form
//
//     P' = W(P) + n_avg * R(P)
//
// where W is a trilinearly-smoothed hash vector field (lattice warp), R is a
// hashed per-patch relief field, and n_avg is the average geometric normal of
// the triangle fan around the welded vertex. All fields are evaluated in GLOBAL
// voxel space from integer cell hashes, so the displacement is bit-identical on
// both sides of a mesh-block boundary. Triangle normals are accumulated over a
// one-cell skirt beyond the block so boundary fans are complete; skirt triangles
// are not emitted.
//
// The mesh is emitted flat-shaded (per-triangle normals, unindexed), with a
// uniform vertex color and triplanar UVs derived from the un-warped surface
// position, matching the source implementation.
//
// LOD is not supported yet (no transition cells, and the stylization fields do
// not currently rescale across LODs) — use with VoxelTerrain.
class VoxelMesherTetra : public VoxelMesher {
	GDCLASS(VoxelMesherTetra, VoxelMesher)

public:
	static const int MIN_SUBDIVISIONS = 1;
	static const int MAX_SUBDIVISIONS = 4;

	VoxelMesherTetra();
	~VoxelMesherTetra();

	void build(VoxelMesher::Output &output, const VoxelMesher::Input &input) override;

	// How many marching cells per voxel along each axis. The source project calls
	// this `island_mesh_resolution_per_voxel` and ships at 2. Density between
	// voxel samples is interpolated trilinearly.
	void set_subdivisions(int p_subdivisions);
	int get_subdivisions() const;

	// Master switch for the stylization pass. Off = raw marching tetrahedra.
	void set_warp_enabled(bool p_enabled);
	bool get_warp_enabled() const;

	// Amplitude, in voxels, of the smoothed hash vector field displacing every
	// surface vertex ("irregular grid" component of the look).
	void set_lattice_warp(float p_amount);
	float get_lattice_warp() const;

	// Amplitude, in voxels, of the along-normal relief applied per hash patch
	// ("faceted patchwork" component of the look).
	void set_patch_relief(float p_amount);
	float get_patch_relief() const;

	// Edge length, in voxels, of the cubic cells the relief field is hashed on.
	void set_patch_size(float p_size);
	float get_patch_size() const;

	// 0 = hard per-patch steps (byte-identical to the source at 0),
	// 1 = fully trilinearly smoothed relief.
	void set_patch_relief_smoothing(float p_smoothing);
	float get_patch_relief_smoothing() const;

	// Uniform vertex color written to every vertex (COLOR array).
	void set_default_color(Color p_color);
	Color get_default_color() const;

	// Scale applied to the triplanar UVs derived from un-warped surface
	// positions (source: `material_sample_uv_scale`).
	void set_uv_scale(float p_scale);
	float get_uv_scale() const;

	// Emit ARRAY_TANGENT (per-triangle tangent from the flat normal, binormal
	// sign +1). Required for normal-mapped PBR materials.
	void set_generate_tangents(bool p_enabled);
	bool get_generate_tangents() const;

	// Output transform baked into emitted vertex positions:
	// final = local * vertex_scale + vertex_offset. Identity by default.
	// Lets build_mesh() callers get world-unit, recentered meshes directly.
	void set_vertex_scale(float p_scale);
	float get_vertex_scale() const;
	void set_vertex_offset(Vector3 p_offset);
	Vector3 get_vertex_offset() const;

	int get_used_channels_mask() const override;

	bool supports_lod() const override {
		return false;
	}

protected:
	static void _bind_methods();

private:
	// All parameters are copied into a local struct at the start of build(), so
	// concurrent builds see one coherent set of values.
	struct Parameters {
		int subdivisions = 2;
		bool warp_enabled = true;
		float lattice_warp = 0.24f;
		float patch_relief = 0.11f;
		float patch_size = 1.25f;
		float patch_relief_smoothing = 0.f;
		Color default_color = Color(1, 1, 1, 1);
		float uv_scale = 0.35f;
		bool generate_tangents = false;
		float vertex_scale = 1.f;
		Vector3 vertex_offset;
	};

	Parameters _parameters;
};

} // namespace zylann::voxel

#endif // VOXEL_MESHER_TETRA_H
