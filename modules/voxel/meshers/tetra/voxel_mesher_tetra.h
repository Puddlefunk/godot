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

	// Carry voxel material through to the mesh, in the format the rest of the
	// module already speaks: CHANNEL_INDICES / CHANNEL_WEIGHTS are read at the
	// solid side of each surface vertex and written to ARRAY_CUSTOM1 as
	// ARRAY_CUSTOM_RG_FLOAT -- x holds four 8-bit indices, y four 8-bit
	// weights, exactly as VoxelMesherTransvoxel packs them.
	//
	// That one format choice buys two consumers, not one. A shader can blend
	// four surfaces per vertex (see doc/source/smooth_terrain.md), and
	// VoxelInstanceGenerator's voxel_material_filter reads the same attribute
	// off ARRAY_CUSTOM1 to decide where a scatter item is allowed to grow --
	// so per-biome foliage falls out of it with no further engine work.
	//
	// Off by default: it costs two channel reads per emitted vertex plus the
	// extra vertex attribute, and a generator that writes no material channels
	// would only produce index 0 everywhere.
	void set_texturing_enabled(bool p_enabled);
	bool get_texturing_enabled() const;

	// Emit ARRAY_TANGENT (per-triangle tangent from the flat normal, binormal
	// sign +1). Required for normal-mapped PBR materials.
	void set_generate_tangents(bool p_enabled);
	bool get_generate_tangents() const;
	void set_lod_transitions_enabled(bool p_enabled);
	bool get_lod_transitions_enabled() const;
	void set_lod_geomorph_enabled(bool p_enabled);
	bool get_lod_geomorph_enabled() const;

	// Output transform baked into emitted vertex positions:
	// final = local * vertex_scale + vertex_offset. Identity by default.
	// Lets build_mesh() callers get world-unit, recentered meshes directly.
	void set_vertex_scale(float p_scale);
	float get_vertex_scale() const;
	void set_vertex_offset(Vector3 p_offset);
	Vector3 get_vertex_offset() const;

	void set_collision_subdivisions(int p_subdivisions);
	int get_collision_subdivisions() const;

	// Where a point on the ANALYTIC isosurface ends up on the VISIBLE one.
	//
	// build() displaces every vertex through stylized_surface_position AFTER
	// the SDF has been sampled, so anything placed by tracing the SDF -- props,
	// scatter, traversal seats -- sits on a surface that is up to
	// lattice_warp (x0.45 on Y) plus patch_relief away from the one the player
	// sees. terrain_sense.gd's header called that "usually invisible"; it is
	// not, for anything with a hard base edge sitting flush on the ground.
	//
	// This is a BINDING, not a second implementation: it calls the same
	// function build() calls, with this mesher's own live parameters, so the
	// two cannot drift. Configure a mesher exactly as the build did (same
	// warp_enabled / lattice_warp / patch_relief / patch_size / smoothing) and
	// the answer is exact.
	//
	// Takes and returns WORLD units; the transform itself works in global voxel
	// units, hence voxel_size. Returns the input unchanged when warp_enabled is
	// false or voxel_size is non-positive, so callers need no special case.
	Vector3 stylize_world_point(Vector3 p_world_position, Vector3 p_world_normal, float p_voxel_size) const;

	int get_used_channels_mask() const override;

	bool supports_lod() const override {
		return true;
	}

	bool is_generating_collision_surface() const override {
		return true;
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
		bool lod_transitions_enabled = true;
		bool lod_geomorph_enabled = false;
		bool texturing_enabled = false;
		float vertex_scale = 1.f;
		Vector3 vertex_offset;
		int collision_subdivisions = 1;
	};

	void build_internal(VoxelMesher::Output &output, const VoxelMesher::Input &input, const Parameters &params);

	Parameters _parameters;
};

} // namespace zylann::voxel

#endif // VOXEL_MESHER_TETRA_H
