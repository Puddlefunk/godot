#include "voxel_mesher_tetra.h"
#include "../../storage/mixel4.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/containers/std_unordered_map.h"
#include "../../util/containers/std_vector.h"
#include "../../util/godot/classes/array_mesh.h"
#include "../../util/godot/classes/rendering_server.h"
#include "../../util/godot/core/class_db.h"
#include "../../util/godot/core/packed_arrays.h"
#include "../../util/profiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// Temporary diagnostics: appends one line per build() stage to a fixed path.
// Remove by setting TETRA_DEBUG_LOG to 0.
#define TETRA_DEBUG_LOG 0
#if TETRA_DEBUG_LOG
#include <cstdio>
#include <cstdarg>
static void tetra_dbg(const char *fmt, ...) {
	FILE *f = fopen("C:/Users/puddles/Claude/Projects/GAME-DESIGN/tetra_debug.log", "ab");
	if (f == nullptr) {
		return;
	}
	va_list args;
	va_start(args, fmt);
	vfprintf(f, fmt, args);
	va_end(args);
	fputc('\n', f);
	fclose(f);
}
#define TETRA_DBG(...) tetra_dbg(__VA_ARGS__)
#else
#define TETRA_DBG(...)
#endif

namespace zylann::voxel {

namespace tetra_mesher {

// ---------------------------------------------------------------------------
// Small double-precision vector. Positions are computed in GLOBAL voxel space
// so that a vertex shared by two mesh blocks is derived from bit-identical
// inputs on both sides; doubles keep that exact until the final, block-local
// float cast.
struct DVec3 {
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

	DVec3() {}
	DVec3(double p_x, double p_y, double p_z) : x(p_x), y(p_y), z(p_z) {}

	inline DVec3 operator+(const DVec3 &o) const {
		return DVec3(x + o.x, y + o.y, z + o.z);
	}
	inline DVec3 operator-(const DVec3 &o) const {
		return DVec3(x - o.x, y - o.y, z - o.z);
	}
	inline DVec3 operator*(double s) const {
		return DVec3(x * s, y * s, z * s);
	}
	inline DVec3 cross(const DVec3 &o) const {
		return DVec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
	}
	inline double dot(const DVec3 &o) const {
		return x * o.x + y * o.y + z * o.z;
	}
	inline double length_squared() const {
		return x * x + y * y + z * z;
	}
	inline DVec3 lerp(const DVec3 &o, double t) const {
		return DVec3(x + (o.x - x) * t, y + (o.y - y) * t, z + (o.z - z) * t);
	}
	inline DVec3 normalized() const {
		const double lsq = length_squared();
		if (lsq <= 0.0) {
			return DVec3();
		}
		const double inv = 1.0 / std::sqrt(lsq);
		return DVec3(x * inv, y * inv, z * inv);
	}
};

inline int64_t floor_to_int(double v) {
	return static_cast<int64_t>(std::floor(v));
}

// ---------------------------------------------------------------------------
// Deterministic hash fields, ported verbatim from the GDScript implementation
// (float math on purpose: the look was tuned against these exact values).

inline float hash_float(int64_t cx, int64_t cy, int64_t cz, int salt) {
	const float n = std::sin(static_cast<float>(cx) * 127.1f + static_cast<float>(cy) * 311.7f +
							 static_cast<float>(cz) * 74.7f + static_cast<float>(salt) * 19.19f) *
			43758.5453123f;
	return n - std::floor(n);
}

inline float hash_signed(int64_t cx, int64_t cy, int64_t cz, int salt) {
	return hash_float(cx, cy, cz, salt) * 2.f - 1.f;
}

inline DVec3 hash_vector(int64_t cx, int64_t cy, int64_t cz) {
	return DVec3(hash_signed(cx, cy, cz, 11), hash_signed(cx, cy, cz, 23) * 0.45f, hash_signed(cx, cy, cz, 37));
}

inline double smoothstep01(double f) {
	return f * f * (3.0 - 2.0 * f);
}

// Trilinear smoothstep blend of the hash vector field. `p` is in global voxel
// units.
DVec3 smooth_lattice_warp(const DVec3 &p) {
	const int64_t bx = floor_to_int(p.x);
	const int64_t by = floor_to_int(p.y);
	const int64_t bz = floor_to_int(p.z);
	const double sx = smoothstep01(p.x - static_cast<double>(bx));
	const double sy = smoothstep01(p.y - static_cast<double>(by));
	const double sz = smoothstep01(p.z - static_cast<double>(bz));

	const DVec3 x00 = hash_vector(bx, by, bz).lerp(hash_vector(bx + 1, by, bz), sx);
	const DVec3 x10 = hash_vector(bx, by + 1, bz).lerp(hash_vector(bx + 1, by + 1, bz), sx);
	const DVec3 x01 = hash_vector(bx, by, bz + 1).lerp(hash_vector(bx + 1, by, bz + 1), sx);
	const DVec3 x11 = hash_vector(bx, by + 1, bz + 1).lerp(hash_vector(bx + 1, by + 1, bz + 1), sx);
	const DVec3 y0 = x00.lerp(x10, sy);
	const DVec3 y1 = x01.lerp(x11, sy);
	return y0.lerp(y1, sz);
}

double smooth_patch_relief(const DVec3 &grid, int64_t bx, int64_t by, int64_t bz) {
	const double sx = smoothstep01(grid.x - static_cast<double>(bx));
	const double sy = smoothstep01(grid.y - static_cast<double>(by));
	const double sz = smoothstep01(grid.z - static_cast<double>(bz));

	const double x00 = Math::lerp(
			static_cast<double>(hash_signed(bx, by, bz, 71)), static_cast<double>(hash_signed(bx + 1, by, bz, 71)), sx);
	const double x10 = Math::lerp(static_cast<double>(hash_signed(bx, by + 1, bz, 71)),
			static_cast<double>(hash_signed(bx + 1, by + 1, bz, 71)), sx);
	const double x01 = Math::lerp(static_cast<double>(hash_signed(bx, by, bz + 1, 71)),
			static_cast<double>(hash_signed(bx + 1, by, bz + 1, 71)), sx);
	const double x11 = Math::lerp(static_cast<double>(hash_signed(bx, by + 1, bz + 1, 71)),
			static_cast<double>(hash_signed(bx + 1, by + 1, bz + 1, 71)), sx);
	const double y0 = Math::lerp(x00, x10, sy);
	const double y1 = Math::lerp(x01, x11, sy);
	return Math::lerp(y0, y1, sz);
}

// `grid` is the surface point expressed in patch units. At smoothing 0 this is
// a plain floor lookup, matching the source byte-for-byte.
double patch_relief(const DVec3 &grid, double smoothing) {
	const int64_t bx = floor_to_int(grid.x);
	const int64_t by = floor_to_int(grid.y);
	const int64_t bz = floor_to_int(grid.z);
	const double hard = hash_signed(bx, by, bz, 71);
	if (smoothing <= 0.0) {
		return hard;
	}
	return Math::lerp(hard, smooth_patch_relief(grid, bx, by, bz), CLAMP(smoothing, 0.0, 1.0));
}

// W(p) + n * R(p). `p` and the return value are in global voxel units.
DVec3 stylized_surface_position(
		const DVec3 &p, const DVec3 &normal, double lattice_warp, double relief_amount, double patch_size,
		double relief_smoothing) {
	DVec3 warped = p;
	if (lattice_warp != 0.0) {
		warped = p + smooth_lattice_warp(p) * lattice_warp;
	}
	if (relief_amount == 0.0) {
		return warped;
	}
	const double inv_patch = 1.0 / MAX(0.1, patch_size);
	const double relief = patch_relief(warped * inv_patch, relief_smoothing) * relief_amount;
	return warped + normal * relief;
}

// ---------------------------------------------------------------------------
// Lattice tables.

// Cube corner offsets, in the same order as the source project.
static const int8_t CUBE_CORNERS[8][3] = {
	{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }
};

// Six positively-oriented tetrahedra (signed volume +1) sharing the 0-6
// diagonal. Winding for every emitted case is decided by permutation parity
// alone -- see polygonise_tetra.
static const int8_t TETRAHEDRA[6][4] = {
	{ 0, 5, 1, 6 }, { 0, 1, 2, 6 }, { 0, 2, 3, 6 }, { 0, 3, 7, 6 }, { 0, 7, 4, 6 }, { 0, 4, 5, 6 }
};

// Parity of a permutation of tetra-local corner slots, by inversion count.
inline bool permutation_is_even(const int slots[4]) {
	int inversions = 0;
	for (int i = 0; i < 4; ++i) {
		for (int j = i + 1; j < 4; ++j) {
			if (slots[i] > slots[j]) {
				++inversions;
			}
		}
	}
	return (inversions % 2) == 0;
}

inline int tet_slot_of(const int8_t tet[4], int corner_index) {
	for (int i = 0; i < 4; ++i) {
		if (tet[i] == corner_index) {
			return i;
		}
	}
	return -1;
}

inline DVec3 edge_interp(const DVec3 &a, const DVec3 &b, double value_a, double value_b) {
	const double divisor = value_a - value_b;
	if (std::abs(divisor) < 0.00001) {
		return a;
	}
	const double t = CLAMP(value_a / divisor, 0.0, 1.0);
	return a.lerp(b, t);
}

// Every marching-tetrahedra vertex lies on an edge between two lattice points,
// so the unordered pair of GLOBAL lattice ids is an exact identity for it:
// welding and warp averaging key on it, and interpolation is canonicalised to
// the lower id so both cells (and both mesh blocks) sharing an edge compute a
// bit-identical position.
inline uint64_t make_edge_key(uint64_t ka, uint64_t kb) {
	const uint64_t lo = MIN(ka, kb);
	const uint64_t hi = MAX(ka, kb);
	// The ids are 63-bit packed global lattice coordinates, so the pair cannot
	// be packed losslessly into 64 bits like the source project's 32-bit flat
	// indices could. Mix them instead (SplitMix64-style): deterministic across
	// blocks, and collisions within one block build are vanishingly unlikely.
	uint64_t h = lo * 0x9E3779B97F4A7C15ull;
	h ^= h >> 32;
	h += hi * 0xBF58476D1CE4E5B9ull;
	h ^= h >> 29;
	return h;
}

struct BuildScratch {
	StdVector<float> densities; // Indexed by local lattice point, sign convention: >= 0 means inside.
	// Timeline slots: every triangle lives on a sub-interval [t_in, t_out] of
	// the geomorph weight. tri_positions holds its geometry at t_in,
	// tri_morph_positions at t_out. Without geomorph there is exactly one
	// interval [0, 1] and both sets are identical.
	StdVector<DVec3> tri_positions; // 3 entries per triangle, global voxel units, at interval START.
	StdVector<DVec3> tri_morph_positions; // Same, at interval END.
	StdVector<uint64_t> tri_keys; // Weld identity per triangle corner.
	StdVector<uint8_t> tri_emit; // 1 entry per triangle: inside the emission window?
	StdVector<uint8_t> tri_transition_on;
	StdVector<uint8_t> tri_transition_off;
	StdVector<float> tri_t_in; // 1 entry per triangle.
	StdVector<float> tri_t_out;

	void clear_mesh() {
		tri_positions.clear();
		tri_morph_positions.clear();
		tri_keys.clear();
		tri_emit.clear();
		tri_transition_on.clear();
		tri_transition_off.clear();
		tri_t_in.clear();
		tri_t_out.clear();
	}
};

BuildScratch &get_tls_scratch() {
	thread_local BuildScratch tls_scratch;
	return tls_scratch;
}

} // namespace tetra_mesher

using namespace tetra_mesher;

VoxelMesherTetra::VoxelMesherTetra() {
	// One extra layer of MAX-side padding so the parent (stride-2) lattice
	// covers the one-cell accumulation skirt. Without it, the skirt on every
	// block's max faces samples the parent field past its last cube
	// (sample_parent_tetra_density's clamp turns into EXTRAPOLATION, f up to
	// 1.5), the morph warp fans built from those values differ from the
	// neighboring block's -- whose min side is properly padded -- and the
	// geomorphed surface steps at the seam. Measured live 2026-08-23 as the
	// long LOD-seam slit.
	set_padding(2, 3);
}

VoxelMesherTetra::~VoxelMesherTetra() {}

int VoxelMesherTetra::get_used_channels_mask() const {
	int mask = (1 << VoxelBuffer::CHANNEL_SDF);
	if (_parameters.texturing_enabled) {
		mask |= (1 << VoxelBuffer::CHANNEL_INDICES) | (1 << VoxelBuffer::CHANNEL_WEIGHTS);
	}
	return mask;
}

void VoxelMesherTetra::build(VoxelMesher::Output &output, const VoxelMesher::Input &input) {
	const Parameters params = _parameters;
	const int render_subdivisions = CLAMP(params.subdivisions, MIN_SUBDIVISIONS, MAX_SUBDIVISIONS);
	const int collision_subdivisions = MIN(
			render_subdivisions, CLAMP(params.collision_subdivisions, MIN_SUBDIVISIONS, MAX_SUBDIVISIONS));
	if (input.collision_hint && collision_subdivisions != render_subdivisions) {
		VoxelMesher::Input render_input = input;
		render_input.collision_hint = false;
		build_internal(output, render_input, params);

		Parameters collision_params = params;
		collision_params.subdivisions = collision_subdivisions;
		VoxelMesher::Output collision_output;
		build_internal(collision_output, input, collision_params);
		output.collision_surface.positions.swap(collision_output.collision_surface.positions);
		output.collision_surface.indices.swap(collision_output.collision_surface.indices);
		return;
	}
	build_internal(output, input, params);
}

void VoxelMesherTetra::build_internal(
		VoxelMesher::Output &output, const VoxelMesher::Input &input, const Parameters &params) {
	ZN_PROFILE_SCOPE();

	const VoxelBuffer &voxels = input.voxels;
	const Vector3i buffer_size = voxels.get_size();
	const int min_pad = get_minimum_padding();
	const int max_pad = get_maximum_padding();

	const Vector3i content_size = buffer_size - Vector3i(min_pad + max_pad, min_pad + max_pad, min_pad + max_pad);
	TETRA_DBG("build: size=(%d,%d,%d) pads=%d/%d subdiv=%d warp=%d", buffer_size.x, buffer_size.y, buffer_size.z,
			min_pad, max_pad, params.subdivisions, int(params.warp_enabled));
	if (content_size.x < 1 || content_size.y < 1 || content_size.z < 1) {
		TETRA_DBG("RET content_size");
		return;
	}

	// Uniform blocks (all air or all matter) cannot contain the isosurface.
	if (voxels.is_uniform(VoxelBuffer::CHANNEL_SDF)) {
		TETRA_DBG("RET uniform");
		return;
	}

	const int r = CLAMP(params.subdivisions, MIN_SUBDIVISIONS, MAX_SUBDIVISIONS);
	const int lod_stride = 1 << input.lod_index;
	const double step = 1.0 / static_cast<double>(r);
	const double world_step = static_cast<double>(lod_stride) * step;
	const Vector3i content_origin = input.origin_in_voxels;
	const Vector3i padded_origin = content_origin - Vector3iUtil::create(min_pad * lod_stride);

	// -----------------------------------------------------------------------
	// Read all SDF samples once, then build the density lattice. Density uses
	// the source project's sign convention: positive = inside.
	StdVector<float> sdf_samples;
	sdf_samples.resize(static_cast<size_t>(buffer_size.x) * buffer_size.y * buffer_size.z);
	{
		float min_sdf = 3.4e38f;
		float max_sdf = -3.4e38f;
		size_t si = 0;
		for (int z = 0; z < buffer_size.z; ++z) {
			for (int y = 0; y < buffer_size.y; ++y) {
				for (int x = 0; x < buffer_size.x; ++x) {
					const float sd = voxels.get_voxel_f(x, y, z, VoxelBuffer::CHANNEL_SDF);
					sdf_samples[si++] = sd;
					min_sdf = MIN(min_sdf, sd);
					max_sdf = MAX(max_sdf, sd);
				}
			}
		}
		if (min_sdf >= 0.f || max_sdf < 0.f) {
			TETRA_DBG("RET sign min=%f max=%f", min_sdf, max_sdf);
			return;
		}
		TETRA_DBG("sdf ok min=%f max=%f", min_sdf, max_sdf);
	}
	const auto sample_index = [&buffer_size](int x, int y, int z) -> size_t {
		return static_cast<size_t>(x) +
				static_cast<size_t>(y) * buffer_size.x +
				static_cast<size_t>(z) * buffer_size.x * buffer_size.y;
	};

	// Lattice of density samples: `r` cells per voxel, spanning every buffer
	// sample. Lattice point l sits at buffer position l/r.
	const Vector3i lattice_points(
			(buffer_size.x - 1) * r + 1, (buffer_size.y - 1) * r + 1, (buffer_size.z - 1) * r + 1);

	BuildScratch &scratch = get_tls_scratch();
	scratch.clear_mesh();
	scratch.densities.resize(static_cast<size_t>(lattice_points.x) * lattice_points.y * lattice_points.z);

	const auto lattice_index = [&lattice_points](int lx, int ly, int lz) -> size_t {
		return static_cast<size_t>(lx) +
				static_cast<size_t>(ly) * lattice_points.x +
				static_cast<size_t>(lz) * lattice_points.x * lattice_points.y;
	};

	{
		ZN_PROFILE_SCOPE_NAMED("Density lattice");
		size_t li = 0;
		for (int lz = 0; lz < lattice_points.z; ++lz) {
			const int z0 = MIN(lz / r, buffer_size.z - 2);
			const double fz = lz * step - z0;
			for (int ly = 0; ly < lattice_points.y; ++ly) {
				const int y0 = MIN(ly / r, buffer_size.y - 2);
				const double fy = ly * step - y0;
				for (int lx = 0; lx < lattice_points.x; ++lx) {
					if (r == 1) {
						scratch.densities[li++] = -sdf_samples[sample_index(lx, ly, lz)];
						continue;
					}
					const int x0 = MIN(lx / r, buffer_size.x - 2);
					const double fx = lx * step - x0;
					// Trilinear interpolation of the SDF between voxel samples.
					const double c000 = sdf_samples[sample_index(x0, y0, z0)];
					const double c100 = sdf_samples[sample_index(x0 + 1, y0, z0)];
					const double c010 = sdf_samples[sample_index(x0, y0 + 1, z0)];
					const double c110 = sdf_samples[sample_index(x0 + 1, y0 + 1, z0)];
					const double c001 = sdf_samples[sample_index(x0, y0, z0 + 1)];
					const double c101 = sdf_samples[sample_index(x0 + 1, y0, z0 + 1)];
					const double c011 = sdf_samples[sample_index(x0, y0 + 1, z0 + 1)];
					const double c111 = sdf_samples[sample_index(x0 + 1, y0 + 1, z0 + 1)];
					const double x00 = c000 + (c100 - c000) * fx;
					const double x10 = c010 + (c110 - c010) * fx;
					const double x01 = c001 + (c101 - c001) * fx;
					const double x11 = c011 + (c111 - c011) * fx;
					const double y0v = x00 + (x10 - x00) * fy;
					const double y1v = x01 + (x11 - x01) * fy;
					scratch.densities[li++] = static_cast<float>(-(y0v + (y1v - y0v) * fz));
				}
			}
		}
	}

	// -----------------------------------------------------------------------
	// March. Emission window covers the block content; the accumulation window
	// adds a one-cell skirt so warp normal fans are complete at block borders
	// (skirt triangles feed normal averaging but are never emitted).
	const int emit_min = min_pad * r;
	const Vector3i emit_max( // Exclusive.
			(min_pad + content_size.x) * r, (min_pad + content_size.y) * r, (min_pad + content_size.z) * r);
	const int acc_min = emit_min - 1;
	const Vector3i acc_max(
			MIN(emit_max.x + 1, lattice_points.x - 1),
			MIN(emit_max.y + 1, lattice_points.y - 1),
			MIN(emit_max.z + 1, lattice_points.z - 1));

	// Global lattice ids use units of 1/subdivisions at LOD0.
	const int64_t axis_bias = int64_t(1) << 20;
	const auto global_lattice_id = [&](int lx, int ly, int lz) -> uint64_t {
		const int64_t gx = static_cast<int64_t>(padded_origin.x) * r + static_cast<int64_t>(lx) * lod_stride + axis_bias;
		const int64_t gy = static_cast<int64_t>(padded_origin.y) * r + static_cast<int64_t>(ly) * lod_stride + axis_bias;
		const int64_t gz = static_cast<int64_t>(padded_origin.z) * r + static_cast<int64_t>(lz) * lod_stride + axis_bias;
		// 21+21+21 bits; unique within a ~2-million-lattice-cell span per axis,
		// and consistent across neighboring blocks, which is what welding and
		// seam determinism require.
		return (static_cast<uint64_t>(gx & 0x1fffff) << 42) | (static_cast<uint64_t>(gy & 0x1fffff) << 21) |
				static_cast<uint64_t>(gz & 0x1fffff);
	};

	const double area_epsilon = 1.0e-8 * world_step * world_step * world_step * world_step;
	// Slot REJECTION uses a far tighter bar than fan/normal hygiene above:
	// area_epsilon scales with step^4, so the same WORLD-SPACE sliver is
	// judged 16x more leniently one LOD up -- the coarse build culled slivers
	// the fine build kept, and the w=1 identity broke by exactly those
	// missing triangles (found 2026-08-23 via the parent-identity test: a
	// 4.8 mm hole where the surface grazes a parent lattice corner). Slivers
	// are load-bearing; only true degeneracy may be dropped.
	const double reject_epsilon = 1.0e-14 * world_step * world_step * world_step * world_step;

	const Vector3i parent_source_size(
			(buffer_size.x + 1) / 2, (buffer_size.y + 1) / 2, (buffer_size.z + 1) / 2);
	const Vector3i parent_lattice_points(
			(parent_source_size.x - 1) * r + 1,
			(parent_source_size.y - 1) * r + 1,
			(parent_source_size.z - 1) * r + 1);
	const auto sample_kuhn_cube = [](const double values[8], const double f[3]) -> double {
		int axes[3] = { 0, 1, 2 };
		if (f[axes[0]] < f[axes[1]]) {
			std::swap(axes[0], axes[1]);
		}
		if (f[axes[1]] < f[axes[2]]) {
			std::swap(axes[1], axes[2]);
		}
		if (f[axes[0]] < f[axes[1]]) {
			std::swap(axes[0], axes[1]);
		}
		const auto corner_from_axes = [](int a, int b) -> int {
			int coords[3] = { 0, 0, 0 };
			if (a >= 0) {
				coords[a] = 1;
			}
			if (b >= 0) {
				coords[b] = 1;
			}
			for (int ci = 0; ci < 8; ++ci) {
				if (CUBE_CORNERS[ci][0] == coords[0] && CUBE_CORNERS[ci][1] == coords[1] &&
						CUBE_CORNERS[ci][2] == coords[2]) {
					return ci;
				}
			}
			return 0;
		};
		const int c1 = corner_from_axes(axes[0], -1);
		const int c2 = corner_from_axes(axes[0], axes[1]);
		return values[0] * (1.0 - f[axes[0]]) + values[c1] * (f[axes[0]] - f[axes[1]]) +
				values[c2] * (f[axes[1]] - f[axes[2]]) + values[6] * f[axes[2]];
	};
	const bool geomorph_enabled = input.lod_hint && params.lod_geomorph_enabled;

	const auto interpolate_parent_lattice_density = [&](int px, int py, int pz) -> double {
		const int x0 = MIN(px / r, parent_source_size.x - 2);
		const int y0 = MIN(py / r, parent_source_size.y - 2);
		const int z0 = MIN(pz / r, parent_source_size.z - 2);
		const double fx = static_cast<double>(px) / r - x0;
		const double fy = static_cast<double>(py) / r - y0;
		const double fz = static_cast<double>(pz) / r - z0;
		const int sx = x0 * 2;
		const int sy = y0 * 2;
		const int sz = z0 * 2;

		double values[8];
		for (int ci = 0; ci < 8; ++ci) {
			values[ci] = -sdf_samples[sample_index(
					sx + CUBE_CORNERS[ci][0] * 2,
					sy + CUBE_CORNERS[ci][1] * 2,
					sz + CUBE_CORNERS[ci][2] * 2)];
		}
		// LAYER 1 of the parent reconstruction: voxels -> parent marching
		// lattice. The parent block's own "Density lattice" pass refines the
		// SDF between voxel samples TRILINEARLY (see the pass above), so this
		// layer must be trilinear too, or the reconstruction targets a surface
		// the parent never actually meshes at subdivisions > 1. Kuhn P1
		// belongs one layer DOWN, inside parent marching cells
		// (sample_parent_tetra_density), where the parent's tetrahedra live.
		// At subdivisions == 1 the two layers coincide and either is exact,
		// which is how a Kuhn-everywhere version validated before it opened
		// LOD-seam cracks at the shipping subdivision count.
		const double x00 = values[0] + (values[1] - values[0]) * fx;
		const double x10 = values[3] + (values[2] - values[3]) * fx;
		const double x01 = values[4] + (values[5] - values[4]) * fx;
		const double x11 = values[7] + (values[6] - values[7]) * fx;
		const double y0v = x00 + (x10 - x00) * fy;
		const double y1v = x01 + (x11 - x01) * fy;
		return y0v + (y1v - y0v) * fz;
	};
	StdVector<float> parent_densities;
	if (input.lod_hint && (params.lod_transitions_enabled || geomorph_enabled)) {
		parent_densities.resize(
				static_cast<size_t>(parent_lattice_points.x) * parent_lattice_points.y * parent_lattice_points.z);
		size_t density_index = 0;
		for (int pz = 0; pz < parent_lattice_points.z; ++pz) {
			for (int py = 0; py < parent_lattice_points.y; ++py) {
				for (int px = 0; px < parent_lattice_points.x; ++px) {
					parent_densities[density_index++] = interpolate_parent_lattice_density(px, py, pz);
				}
			}
		}
	}
	const auto sample_parent_lattice_density = [&](int px, int py, int pz) -> double {
		const size_t i = static_cast<size_t>(px) + static_cast<size_t>(py) * parent_lattice_points.x +
				static_cast<size_t>(pz) * parent_lattice_points.x * parent_lattice_points.y;
		return parent_densities[i];
	};

	const auto sample_parent_tetra_density = [&](double lx, double ly, double lz) -> double {
		const double fine_coords[3] = { lx, ly, lz };
		const int parent_sizes[3] = { parent_lattice_points.x, parent_lattice_points.y, parent_lattice_points.z };
		int base[3];
		double f[3];
		for (int axis = 0; axis < 3; ++axis) {
			const double q = fine_coords[axis] * 0.5;
			base[axis] = MIN(static_cast<int>(std::floor(q)), parent_sizes[axis] - 2);
			f[axis] = q - base[axis];
		}

		double values[8];
		for (int ci = 0; ci < 8; ++ci) {
			values[ci] = sample_parent_lattice_density(
					base[0] + CUBE_CORNERS[ci][0],
					base[1] + CUBE_CORNERS[ci][1],
					base[2] + CUBE_CORNERS[ci][2]);
		}

		return sample_kuhn_cube(values, f);
	};

	// Parent P1 field sampled once per FINE lattice point. Two wins: the march
	// stops paying up to 8 Kuhn interpolations per cell for EVERY cell in the
	// accumulation window (they used to run before the surface early-out, so
	// mostly on empty cells), and every consumer below -- geomorph targets and
	// transition corners alike -- reads the exact same value per lattice
	// point.
	StdVector<float> fine_parent_densities;
	if (!parent_densities.empty()) {
		ZN_PROFILE_SCOPE_NAMED("Parent fine lattice");
		fine_parent_densities.resize(scratch.densities.size());
		size_t fpd_index = 0;
		for (int lz = 0; lz < lattice_points.z; ++lz) {
			for (int ly = 0; ly < lattice_points.y; ++ly) {
				for (int lx = 0; lx < lattice_points.x; ++lx) {
					fine_parent_densities[fpd_index++] =
							static_cast<float>(sample_parent_tetra_density(lx, ly, lz));
				}
			}
		}
	}
	const bool parent_field_available = !fine_parent_densities.empty();

	{
		ZN_PROFILE_SCOPE_NAMED("March");

		DVec3 corner_positions[8];
		double base_values[8]; // Fine field, after transition substitution: the field at w=0.
		double parent_values[8]; // Parent field: the field at w=1.
		double start_values[8]; // Field at the current interval's start time.
		double end_values[8]; // Field at the current interval's end time.
		double case_values[8]; // Field at the current interval's midpoint -- the sign source.
		uint64_t corner_keys[8];
		uint8_t corner_border_masks[8];
		int inside[4];
		int outside[4];
		int slots[4];
		uint8_t transition_on = 0;
		uint8_t transition_off = 0;
		float interval_t_in = 0.f;
		float interval_t_out = 1.f;

		// ---------------------------------------------------------------
		// THE TIMELINE (2026-08-23): the morphed field at a lattice vertex is
		// the lerp f(t) = (1-t)*f0 + t*f1, so each vertex changes sign at most
		// once, at t* = f0/(f0-f1), and a tetrahedron passes through at most 5
		// marching cases as t goes 0 -> 1 -- the intervals between its
		// corners' flip times. Each interval's triangles are emitted as their
		// own slots carrying [t_in, t_out] plus geometry evaluated at both
		// ends; the vertex shader collapses a slot outside its window and
		// lerps inside it. Births and deaths therefore happen at their TRUE
		// times (the geometry is continuous through each event: a dying
		// triangle's crossings reach the flipping corner exactly at t*, where
		// the next interval's triangles start), and w=0 / w=1 reproduce the
		// fine and parent surfaces exactly. The only approximation left is
		// linear vertex motion WITHIN an interval (the true crossing motion is
		// rational in t) -- bounded, monotone, and endpoint-exact.
		// Blend shapes cannot express any of this; they are no longer used.
		// ---------------------------------------------------------------

		const auto emit_edge_vertex =
				[&](int i, int j, DVec3 &out_start, DVec3 &out_end, uint64_t &out_key) {
					int lo = i;
					int hi = j;
					if (corner_keys[j] < corner_keys[i]) {
						lo = j;
						hi = i;
					}
					out_start = edge_interp(
							corner_positions[lo], corner_positions[hi], start_values[lo], start_values[hi]);
					out_end = edge_interp(
							corner_positions[lo], corner_positions[hi], end_values[lo], end_values[hi]);
					out_key = make_edge_key(corner_keys[i], corner_keys[j]);
					const uint8_t modified_faces = (corner_border_masks[i] | corner_border_masks[j]) & transition_on;
					if (modified_faces != 0) {
						out_key ^= static_cast<uint64_t>(modified_faces) * 0xD6E8FEB86659FD93ull;
						out_key ^= out_key >> 31;
					}
				};

		const auto add_triangle = [&](const DVec3 &a_start, const DVec3 &a_end, uint64_t key_a, //
										  const DVec3 &b_start, const DVec3 &b_end, uint64_t key_b, //
										  const DVec3 &c_start, const DVec3 &c_end, uint64_t key_c, bool emit) {
			// A slot earns its place with area at EITHER end of its interval:
			// births are degenerate at t_in by construction, deaths at t_out.
			// Only a triangle degenerate at both ends carries nothing.
			const bool start_has_area =
					(b_start - a_start).cross(c_start - a_start).length_squared() >= reject_epsilon;
			const bool end_has_area = (b_end - a_end).cross(c_end - a_end).length_squared() >= reject_epsilon;
			if (!start_has_area && !end_has_area) {
				return;
			}
			scratch.tri_positions.push_back(a_start);
			scratch.tri_positions.push_back(b_start);
			scratch.tri_positions.push_back(c_start);
			scratch.tri_morph_positions.push_back(a_end);
			scratch.tri_morph_positions.push_back(b_end);
			scratch.tri_morph_positions.push_back(c_end);
			scratch.tri_keys.push_back(key_a);
			scratch.tri_keys.push_back(key_b);
			scratch.tri_keys.push_back(key_c);
			scratch.tri_emit.push_back(emit ? 1 : 0);
			scratch.tri_transition_on.push_back(transition_on);
			scratch.tri_transition_off.push_back(transition_off);
			scratch.tri_t_in.push_back(interval_t_in);
			scratch.tri_t_out.push_back(interval_t_out);
		};

		// One marching case, one interval. Signs come from case_values
		// (interval midpoint, where no corner is at zero); positions come from
		// start_values / end_values.
		const auto polygonise_tet_interval = [&](const int8_t *tet, bool emit) {
			int inside_count = 0;
			int outside_count = 0;
			for (int k = 0; k < 4; ++k) {
				const int corner_index = tet[k];
				if (case_values[corner_index] >= 0.0) {
					inside[inside_count++] = corner_index;
				} else {
					outside[outside_count++] = corner_index;
				}
			}
			if (inside_count == 0 || inside_count == 4) {
				return;
			}

			if (inside_count == 1) {
				const int a = inside[0];
				slots[0] = tet_slot_of(tet, a);
				slots[1] = tet_slot_of(tet, outside[0]);
				slots[2] = tet_slot_of(tet, outside[1]);
				slots[3] = tet_slot_of(tet, outside[2]);
				DVec3 p0s, p1s, p2s, p0e, p1e, p2e;
				uint64_t k0, k1, k2;
				emit_edge_vertex(a, outside[0], p0s, p0e, k0);
				emit_edge_vertex(a, outside[1], p1s, p1e, k1);
				emit_edge_vertex(a, outside[2], p2s, p2e, k2);
				if (permutation_is_even(slots)) {
					add_triangle(p0s, p0e, k0, p1s, p1e, k1, p2s, p2e, k2, emit);
				} else {
					add_triangle(p0s, p0e, k0, p2s, p2e, k2, p1s, p1e, k1, emit);
				}
			} else if (inside_count == 3) {
				const int a = outside[0];
				slots[0] = tet_slot_of(tet, a);
				slots[1] = tet_slot_of(tet, inside[0]);
				slots[2] = tet_slot_of(tet, inside[1]);
				slots[3] = tet_slot_of(tet, inside[2]);
				DVec3 q0s, q1s, q2s, q0e, q1e, q2e;
				uint64_t j0, j1, j2;
				emit_edge_vertex(a, inside[2], q0s, q0e, j0);
				emit_edge_vertex(a, inside[1], q1s, q1e, j1);
				emit_edge_vertex(a, inside[0], q2s, q2e, j2);
				if (permutation_is_even(slots)) {
					add_triangle(q0s, q0e, j0, q1s, q1e, j1, q2s, q2e, j2, emit);
				} else {
					add_triangle(q0s, q0e, j0, q2s, q2e, j2, q1s, q1e, j1, emit);
				}
			} else {
				const int a = inside[0];
				const int b = inside[1];
				const int c = outside[0];
				const int d = outside[1];
				slots[0] = tet_slot_of(tet, a);
				slots[1] = tet_slot_of(tet, b);
				slots[2] = tet_slot_of(tet, c);
				slots[3] = tet_slot_of(tet, d);
				DVec3 acs, ads, bcs, bds, ace, ade, bce, bde;
				uint64_t ac_key, ad_key, bc_key, bd_key;
				emit_edge_vertex(a, c, acs, ace, ac_key);
				emit_edge_vertex(a, d, ads, ade, ad_key);
				emit_edge_vertex(b, c, bcs, bce, bc_key);
				emit_edge_vertex(b, d, bds, bde, bd_key);
				if (permutation_is_even(slots)) {
					add_triangle(acs, ace, ac_key, bds, bde, bd_key, bcs, bce, bc_key, emit);
					add_triangle(acs, ace, ac_key, ads, ade, ad_key, bds, bde, bd_key, emit);
				} else {
					add_triangle(acs, ace, ac_key, bcs, bce, bc_key, bds, bde, bd_key, emit);
					add_triangle(acs, ace, ac_key, bds, bde, bd_key, ads, ade, ad_key, emit);
				}
			}
		};

		// Marches every tetrahedron of the current cell through its full
		// timeline. base_values/parent_values must hold the w=0 / w=1 fields
		// (after any transition substitution); without geomorph they are
		// identical and this degenerates to exactly one interval [0, 1].
		const auto polygonise_cell = [&](bool emit) {
			for (int ti = 0; ti < 6; ++ti) {
				const int8_t *tet = TETRAHEDRA[ti];

				// Distinct flip times of this tet's corners, sorted.
				double flips[4];
				int flip_count = 0;
				for (int k = 0; k < 4; ++k) {
					const int ci = tet[k];
					const double f0 = base_values[ci];
					const double f1 = parent_values[ci];
					if ((f0 >= 0.0) == (f1 >= 0.0)) {
						continue;
					}
					const double denom = f0 - f1;
					if (std::abs(denom) < 1.0e-12) {
						continue;
					}
					const double t_flip = CLAMP(f0 / denom, 0.0, 1.0);
					bool duplicate = false;
					for (int fi = 0; fi < flip_count; ++fi) {
						if (std::abs(flips[fi] - t_flip) < 1.0e-6) {
							duplicate = true;
							break;
						}
					}
					if (!duplicate) {
						int at = flip_count++;
						while (at > 0 && flips[at - 1] > t_flip) {
							flips[at] = flips[at - 1];
							--at;
						}
						flips[at] = t_flip;
					}
				}

				double t_prev = 0.0;
				for (int fi = 0; fi <= flip_count; ++fi) {
					const double t_next = fi < flip_count ? flips[fi] : 1.0;
					if (t_next - t_prev < 1.0e-6) {
						t_prev = t_next;
						continue;
					}
					const double t_mid = 0.5 * (t_prev + t_next);
					for (int ci = 0; ci < 8; ++ci) {
						const double f0 = base_values[ci];
						const double f1 = parent_values[ci];
						start_values[ci] = f0 + (f1 - f0) * t_prev;
						end_values[ci] = f0 + (f1 - f0) * t_next;
						case_values[ci] = f0 + (f1 - f0) * t_mid;
					}
					interval_t_in = static_cast<float>(t_prev);
					interval_t_out = static_cast<float>(t_next);
					polygonise_tet_interval(tet, emit);
					t_prev = t_next;
				}
			}
		};

		for (int cx = acc_min; cx < acc_max.x; ++cx) {
			for (int cy = acc_min; cy < acc_max.y; ++cy) {
				for (int cz = acc_min; cz < acc_max.z; ++cz) {
					// Originals, preserved across transition subsets (which
					// substitute into base_values/parent_values per pass).
					double cell_fine[8];
					double cell_parent[8];
					bool any_inside = false;
					bool any_outside = false;
					for (int ci = 0; ci < 8; ++ci) {
						const int lx = cx + CUBE_CORNERS[ci][0];
						const int ly = cy + CUBE_CORNERS[ci][1];
						const int lz = cz + CUBE_CORNERS[ci][2];
						const double value = scratch.densities[lattice_index(lx, ly, lz)];
						cell_fine[ci] = value;
						cell_parent[ci] = parent_field_available
								? static_cast<double>(fine_parent_densities[lattice_index(lx, ly, lz)])
								: value;
						if (value >= 0.0) {
							any_inside = true;
						} else {
							any_outside = true;
						}
					}
					const bool emit = cx >= emit_min && cy >= emit_min && cz >= emit_min && cx < emit_max.x &&
							cy < emit_max.y && cz < emit_max.z;
					uint8_t cell_border_mask = 0;
					if (emit) {
						cell_border_mask |= cx == emit_min ? 1 : 0;
						cell_border_mask |= cx == emit_max.x - 1 ? 2 : 0;
						cell_border_mask |= cy == emit_min ? 4 : 0;
						cell_border_mask |= cy == emit_max.y - 1 ? 8 : 0;
						cell_border_mask |= cz == emit_min ? 16 : 0;
						cell_border_mask |= cz == emit_max.z - 1 ? 32 : 0;
					}

					for (int ci = 0; ci < 8; ++ci) {
						uint8_t mask = 0;
						mask |= (cell_border_mask & 1) && CUBE_CORNERS[ci][0] == 0 ? 1 : 0;
						mask |= (cell_border_mask & 2) && CUBE_CORNERS[ci][0] == 1 ? 2 : 0;
						mask |= (cell_border_mask & 4) && CUBE_CORNERS[ci][1] == 0 ? 4 : 0;
						mask |= (cell_border_mask & 8) && CUBE_CORNERS[ci][1] == 1 ? 8 : 0;
						mask |= (cell_border_mask & 16) && CUBE_CORNERS[ci][2] == 0 ? 16 : 0;
						mask |= (cell_border_mask & 32) && CUBE_CORNERS[ci][2] == 1 ? 32 : 0;
						corner_border_masks[ci] = mask;
					}

					const bool transition_cell = input.lod_hint && params.lod_transitions_enabled && cell_border_mask != 0;
					if (transition_cell || geomorph_enabled) {
						// A cell can carry surface at ANY point of the
						// timeline: birth slots exist wherever the PARENT
						// field crosses, even where the fine field does not.
						bool possible_inside = false;
						bool possible_outside = false;
						for (int ci = 0; ci < 8; ++ci) {
							possible_inside |= cell_fine[ci] >= 0.0 || cell_parent[ci] >= 0.0;
							possible_outside |= cell_fine[ci] < 0.0 || cell_parent[ci] < 0.0;
						}
						if (!(possible_inside && possible_outside)) {
							continue;
						}
					} else if (!(any_inside && any_outside)) {
						continue;
					}

					for (int ci = 0; ci < 8; ++ci) {
						const int lx = cx + CUBE_CORNERS[ci][0];
						const int ly = cy + CUBE_CORNERS[ci][1];
						const int lz = cz + CUBE_CORNERS[ci][2];
						corner_keys[ci] = global_lattice_id(lx, ly, lz);
						corner_positions[ci] = DVec3(
								static_cast<double>(padded_origin.x) + lx * world_step,
								static_cast<double>(padded_origin.y) + ly * world_step,
								static_cast<double>(padded_origin.z) + lz * world_step);
					}

					if (transition_cell) {
						uint8_t subset = cell_border_mask;
						for (;;) {
							transition_on = subset;
							transition_off = cell_border_mask & ~subset;
							for (int ci = 0; ci < 8; ++ci) {
								// Corners stitched to the coarse neighbor take
								// the parent value as their w=0 field; with
								// geomorph they also keep it at w=1, so
								// stitched geometry never moves.
								const double stitched_fine =
										(corner_border_masks[ci] & subset) != 0 ? cell_parent[ci] : cell_fine[ci];
								base_values[ci] = stitched_fine;
								parent_values[ci] = geomorph_enabled ? cell_parent[ci] : stitched_fine;
							}
							polygonise_cell(true);
							if (subset == 0) {
								break;
							}
							subset = (subset - 1) & cell_border_mask;
						}
					} else {
						transition_on = 0;
						transition_off = 0;
						for (int ci = 0; ci < 8; ++ci) {
							corner_border_masks[ci] = 0;
							base_values[ci] = cell_fine[ci];
							parent_values[ci] = geomorph_enabled ? cell_parent[ci] : cell_fine[ci];
						}
						polygonise_cell(emit);
					}
				}
			}
		}
	}

	const size_t triangle_count = scratch.tri_positions.size() / 3;
	TETRA_DBG("march tris=%d", int(triangle_count));
	if (triangle_count == 0) {
		TETRA_DBG("RET no tris");
		return;
	}

	// -----------------------------------------------------------------------
	// Stylization: two fan-normal sums per welded edge key -- one accumulated
	// from the geometry alive at t=0 (all slots with t_in == 0, skirt
	// included), one from the geometry alive at t=1 (t_out == 1). The warp is
	// evaluated PER EMITTED VERTEX at each of its interval endpoints, with the
	// normal nlerp-ed between the two fans by that endpoint's time: continuous
	// across interval handoffs (same key + same time on both sides), and
	// seam-consistent (both fans are). W and R depend only on position, so
	// deferring evaluation to averaged normals stays exact.
	StdUnorderedMap<uint64_t, DVec3> fine_normal_sums;
	StdUnorderedMap<uint64_t, DVec3> target_normal_sums;
	if (params.warp_enabled) {
		ZN_PROFILE_SCOPE_NAMED("Warp");
		fine_normal_sums.reserve(triangle_count * 2);
		if (geomorph_enabled) {
			target_normal_sums.reserve(triangle_count * 2);
		}
		for (size_t t = 0; t < triangle_count; ++t) {
			const size_t i = t * 3;
			DVec3 start_normal = (scratch.tri_positions[i + 1] - scratch.tri_positions[i])
						.cross(scratch.tri_positions[i + 2] - scratch.tri_positions[i]);
			const bool start_ok = start_normal.length_squared() >= area_epsilon;
			if (start_ok) {
				start_normal = start_normal.normalized();
			}
			if (scratch.tri_t_in[t] == 0.f && start_ok) {
				for (int corner = 0; corner < 3; ++corner) {
					const uint64_t key = scratch.tri_keys[i + corner];
					auto it = fine_normal_sums.find(key);
					if (it == fine_normal_sums.end()) {
						fine_normal_sums.insert({ key, start_normal });
					} else {
						it->second = it->second + start_normal;
					}
				}
			}
			if (geomorph_enabled && scratch.tri_t_out[t] == 1.f) {
				DVec3 end_normal = (scratch.tri_morph_positions[i + 1] - scratch.tri_morph_positions[i])
							.cross(scratch.tri_morph_positions[i + 2] - scratch.tri_morph_positions[i]);
				if (end_normal.length_squared() >= area_epsilon) {
					end_normal = end_normal.normalized();
					if (start_ok && end_normal.dot(start_normal) < 0.0) {
						end_normal = end_normal * -1.0;
					}
				} else if (start_ok) {
					// Degenerate-at-1 sliver: the fine fan normal stands in so
					// the key still exists on both sides of every seam.
					end_normal = start_normal;
				} else {
					continue;
				}
				for (int corner = 0; corner < 3; ++corner) {
					const uint64_t key = scratch.tri_keys[i + corner];
					auto it = target_normal_sums.find(key);
					if (it == target_normal_sums.end()) {
						target_normal_sums.insert({ key, end_normal });
					} else {
						it->second = it->second + end_normal;
					}
				}
			}
		}
	}

	// Warp displacement for one welded vertex at one timeline position.
	const auto displaced_point = [&](uint64_t key, const DVec3 &raw, double t) -> DVec3 {
		if (!params.warp_enabled) {
			return raw;
		}
		const auto it0 = fine_normal_sums.find(key);
		const auto it1 = target_normal_sums.find(key);
		const bool has0 = it0 != fine_normal_sums.end();
		const bool has1 = it1 != target_normal_sums.end();
		if (!has0 && !has1) {
			return raw;
		}
		const DVec3 n0 = (has0 ? it0->second : it1->second).normalized();
		const DVec3 n1 = (has1 ? it1->second : it0->second).normalized();
		DVec3 n = n0.lerp(n1, t);
		if (n.length_squared() < 1.0e-12) {
			n = n0;
		} else {
			n = n.normalized();
		}
		return stylized_surface_position(
				raw, n, params.lattice_warp, params.patch_relief, params.patch_size, params.patch_relief_smoothing);
	};

	// -----------------------------------------------------------------------
	// Emit: block-local, flat-shaded, unindexed.
	PackedVector3Array out_vertices;
	PackedVector3Array out_normals;
	PackedColorArray out_colors;
	PackedVector2Array out_uvs;
	PackedFloat32Array out_tangents;
	PackedFloat32Array out_lod_attribs;
	PackedFloat32Array out_tex_attribs;
	// CUSTOM2, 4 floats per vertex: xyz = position at the vertex's interval
	// END (t_out), w = its window packed as round(t_in*4095)*4096 +
	// round(t_out*4095) -- 24 bits, exact in float32, and never 0.0 for a
	// real window (t_out=1 encodes 4095), so shaders can treat 0.0 as "no
	// timeline attribute, static vertex".
	PackedFloat32Array out_timeline_attribs;
	StdUnorderedMap<uint64_t, int> collision_vertex_indices;

	{
		ZN_PROFILE_SCOPE_NAMED("Emit");

		// Mesh space: content voxel (0,0,0) at the origin of the block node,
		// then scaled and offset by the user-configured output transform
		// (identity by default; used by callers that bake voxel size and a
		// recentering offset directly into the mesh).
		const DVec3 local_offset(
				static_cast<double>(content_origin.x),
				static_cast<double>(content_origin.y),
				static_cast<double>(content_origin.z));
		const double v_scale = params.vertex_scale;
		const DVec3 v_offset(params.vertex_offset.x, params.vertex_offset.y, params.vertex_offset.z);

		const auto triplanar_uv = [&](const DVec3 &p, const DVec3 &n) -> Vector2 {
			const double ax = std::abs(n.x);
			const double ay = std::abs(n.y);
			const double az = std::abs(n.z);
			if (ay >= ax && ay >= az) {
				return Vector2(p.x, p.z) * params.uv_scale;
			}
			if (ax >= ay && ax >= az) {
				return Vector2(p.z, p.y) * params.uv_scale;
			}
			return Vector2(p.x, p.y) * params.uv_scale;
		};

		// Voxel material for one surface vertex, packed the way the rest of
		// the module expects to read it.
		//
		// The vertex sits ON the isosurface, so the voxel it falls in is as
		// likely to be air as ground -- and an air voxel's material is
		// whatever the generator happened to leave there. Stepping half a
		// voxel back along the surface normal lands on the solid side every
		// time, which is the sample that actually means something.
		//
		const auto sample_tex_attrib = [&](const DVec3 &surface_pos, const DVec3 &normal, float &out_indices_f,
										   float &out_weights_f) {
			const DVec3 inside = surface_pos - normal * (0.5 * lod_stride);
			const double inv_stride = 1.0 / static_cast<double>(lod_stride);
			const int bx = CLAMP(static_cast<int>(std::floor((inside.x - padded_origin.x) * inv_stride + 0.5)), 0,
					buffer_size.x - 1);
			const int by = CLAMP(static_cast<int>(std::floor((inside.y - padded_origin.y) * inv_stride + 0.5)), 0,
					buffer_size.y - 1);
			const int bz = CLAMP(static_cast<int>(std::floor((inside.z - padded_origin.z) * inv_stride + 0.5)), 0,
					buffer_size.z - 1);

			// 4x4-bit in the buffer, 4x8-bit in the mesh. Both defaults are
			// sane on an untouched buffer -- indices (0,1,2,3), weights
			// (255,0,0,0) -- so a generator that writes no material at all
			// still produces a valid "all texture 0" attribute rather than
			// four zero weights the shader would have to divide by.
			const uint16_t packed_indices_u16 =
					static_cast<uint16_t>(voxels.get_voxel(bx, by, bz, VoxelBuffer::CHANNEL_INDICES));
			const uint16_t packed_weights_u16 =
					static_cast<uint16_t>(voxels.get_voxel(bx, by, bz, VoxelBuffer::CHANNEL_WEIGHTS));
			const FixedArray<uint8_t, 4> ind = mixel4::decode_indices_from_packed_u16(packed_indices_u16);
			const FixedArray<uint8_t, 4> wei = mixel4::decode_weights_from_packed_u16(packed_weights_u16);

			const uint32_t pi = static_cast<uint32_t>(ind[0]) | (static_cast<uint32_t>(ind[1]) << 8) |
					(static_cast<uint32_t>(ind[2]) << 16) | (static_cast<uint32_t>(ind[3]) << 24);
			const uint32_t pw = static_cast<uint32_t>(wei[0]) | (static_cast<uint32_t>(wei[1]) << 8) |
					(static_cast<uint32_t>(wei[2]) << 16) | (static_cast<uint32_t>(wei[3]) << 24);
			memcpy(&out_indices_f, &pi, sizeof(float));
			memcpy(&out_weights_f, &pw, sizeof(float));
		};

		const size_t max_vertices = triangle_count * 3;
		out_vertices.resize(max_vertices);
		out_normals.resize(max_vertices);
		out_colors.resize(max_vertices);
		out_uvs.resize(max_vertices);
		if (params.generate_tangents) {
			out_tangents.resize(max_vertices * 4);
		}
		if (input.lod_hint) {
			out_lod_attribs.resize(max_vertices * 4);
		}
		if (params.texturing_enabled) {
			out_tex_attribs.resize(max_vertices * 2);
		}
		if (geomorph_enabled) {
			out_timeline_attribs.resize(max_vertices * 4);
		}
		float *lod_attribs_w = input.lod_hint ? out_lod_attribs.ptrw() : nullptr;
		float *tex_attribs_w = params.texturing_enabled ? out_tex_attribs.ptrw() : nullptr;
		float *timeline_w = geomorph_enabled ? out_timeline_attribs.ptrw() : nullptr;
		Vector3 *vertices_w = out_vertices.ptrw();
		Vector3 *normals_w = out_normals.ptrw();
		Color *colors_w = out_colors.ptrw();
		Vector2 *uvs_w = out_uvs.ptrw();
		float *tangents_w = params.generate_tangents ? out_tangents.ptrw() : nullptr;
		size_t vertex_count = 0;
		if (input.collision_hint) {
			collision_vertex_indices.reserve(max_vertices / 3);
		}

		for (size_t t = 0; t < triangle_count; ++t) {
			if (scratch.tri_emit[t] == 0) {
				continue;
			}
			const size_t i = t * 3;
			DVec3 start_a = scratch.tri_positions[i];
			DVec3 start_b = scratch.tri_positions[i + 1];
			DVec3 start_c = scratch.tri_positions[i + 2];
			DVec3 end_a = scratch.tri_morph_positions[i];
			DVec3 end_b = scratch.tri_morph_positions[i + 1];
			DVec3 end_c = scratch.tri_morph_positions[i + 2];
			uint64_t key_a = scratch.tri_keys[i];
			uint64_t key_b = scratch.tri_keys[i + 1];
			uint64_t key_c = scratch.tri_keys[i + 2];
			const double t_in = scratch.tri_t_in[t];
			const double t_out = scratch.tri_t_out[t];

			DVec3 start_source_normal = (start_b - start_a).cross(start_c - start_a);
			DVec3 end_source_normal = (end_b - end_a).cross(end_c - end_a);
			const bool start_has_area = start_source_normal.length_squared() >= reject_epsilon;
			const bool end_has_area = end_source_normal.length_squared() >= reject_epsilon;
			if (!start_has_area && !end_has_area) {
				continue;
			}
			if (start_has_area) {
				start_source_normal = start_source_normal.normalized();
			}
			if (end_has_area) {
				end_source_normal = end_source_normal.normalized();
				if (start_has_area && end_source_normal.dot(start_source_normal) < 0.0) {
					end_source_normal = end_source_normal * -1.0;
				}
			}
			if (!start_has_area) {
				start_source_normal = end_source_normal;
			}
			if (!end_has_area) {
				end_source_normal = start_source_normal;
			}

			DVec3 warped_start_a = displaced_point(key_a, start_a, t_in) - local_offset;
			DVec3 warped_start_b = displaced_point(key_b, start_b, t_in) - local_offset;
			DVec3 warped_start_c = displaced_point(key_c, start_c, t_in) - local_offset;
			DVec3 warped_end_a = displaced_point(key_a, end_a, t_out) - local_offset;
			DVec3 warped_end_b = displaced_point(key_b, end_b, t_out) - local_offset;
			DVec3 warped_end_c = displaced_point(key_c, end_c, t_out) - local_offset;

			DVec3 start_normal = (warped_start_b - warped_start_a).cross(warped_start_c - warped_start_a);
			DVec3 end_normal = (warped_end_b - warped_end_a).cross(warped_end_c - warped_end_a);
			const bool warped_start_ok = start_normal.length_squared() >= reject_epsilon;
			const bool warped_end_ok = end_normal.length_squared() >= reject_epsilon;
			if (!warped_start_ok && !warped_end_ok) {
				continue;
			}
			if (warped_start_ok) {
				start_normal = start_normal.normalized();
			}
			if (warped_end_ok) {
				end_normal = end_normal.normalized();
			}
			// Guard only against the warp itself folding a triangle inside
			// out, judged at whichever end of the interval has area.
			const DVec3 reference_warped = warped_start_ok ? start_normal : end_normal;
			const DVec3 reference_source = warped_start_ok ? start_source_normal : end_source_normal;
			if (reference_warped.dot(reference_source) < 0.0) {
				std::swap(warped_start_b, warped_start_c);
				std::swap(warped_end_b, warped_end_c);
				std::swap(start_b, start_c);
				std::swap(end_b, end_c);
				std::swap(key_b, key_c);
				start_normal = start_normal * -1.0;
				end_normal = end_normal * -1.0;
			}
			if (!warped_start_ok) {
				start_normal = end_normal;
			} else if (!warped_end_ok) {
				end_normal = start_normal;
			} else if (end_normal.dot(start_normal) < 0.0) {
				end_normal = end_normal * -1.0;
			}

			const Vector3 gd_normal(
					static_cast<real_t>(start_normal.x),
					static_cast<real_t>(start_normal.y),
					static_cast<real_t>(start_normal.z));
			const Vector3 gd_end_normal(
					static_cast<real_t>(end_normal.x),
					static_cast<real_t>(end_normal.y),
					static_cast<real_t>(end_normal.z));
			float packed_window = 0.f;
			if (timeline_w != nullptr) {
				const uint32_t t_in_code = static_cast<uint32_t>(std::lround(t_in * 4095.0));
				uint32_t t_out_code = static_cast<uint32_t>(std::lround(t_out * 4095.0));
				if (t_out_code == 0u) {
					t_out_code = 1u;
				}
				packed_window = static_cast<float>(t_in_code * 4096u + t_out_code);
			}

			// Tangent frame aligned to the dominant-axis texture projection,
			// INCLUDING handedness. Godot's BaseMaterial3D triplanar samples
			// (object-space pos, y negated): Z-dominant -> (x, -y),
			// Y-dominant -> (x, z), X-dominant -> (-z, -y). The tangent follows
			// each projection's U axis, and the binormal sign `w` is chosen so
			// B = cross(N, T) * w follows its V axis. The sign alternates with
			// the facing direction of the dominant axis, which is what a constant
			// w gets wrong: half the faces sample a mirrored normal map, and
			// high-contrast maps glint where they should shadow.
			// Tangents are built for the interval START; morph-time tangent
			// rotation is not modeled (the demo's terrain shaders are
			// tangent-free).
			DVec3 tangent;
			float tangent_w = 1.f;
			if (tangents_w != nullptr) {
				const double ax = std::abs(start_normal.x);
				const double ay = std::abs(start_normal.y);
				const double az = std::abs(start_normal.z);
				DVec3 u_axis;
				DVec3 v_axis;
				if (ay >= ax && ay >= az) {
					u_axis = DVec3(1.0, 0.0, 0.0);
					v_axis = DVec3(0.0, 0.0, 1.0);
				} else if (ax >= ay && ax >= az) {
					u_axis = DVec3(0.0, 0.0, -1.0);
					v_axis = DVec3(0.0, -1.0, 0.0);
				} else {
					u_axis = DVec3(1.0, 0.0, 0.0);
					v_axis = DVec3(0.0, -1.0, 0.0);
				}
				tangent = (u_axis - start_normal * u_axis.dot(start_normal)).normalized();
				tangent_w = start_normal.cross(tangent).dot(v_axis) >= 0.0 ? 1.f : -1.f;
			}

			// The marching step emits the mathematical convention
			// (counter-clockwise seen from outside); Godot's rasterizer treats
			// the opposite handedness as front-facing, so corners b and c swap
			// on the way out. The normal stays outward regardless.
			const DVec3 *emit_start[3] = { &warped_start_a, &warped_start_c, &warped_start_b };
			const DVec3 *emit_end[3] = { &warped_end_a, &warped_end_c, &warped_end_b };
			// UVs and material sampling use UN-warped positions (the material
			// belongs to where the surface mathematically is, not where the
			// stylization pushed it), taken from whichever interval end has
			// area -- a birth is degenerate at its start.
			const DVec3 *sample_positions[3] = { start_has_area ? &start_a : &end_a,
				start_has_area ? &start_c : &end_c, start_has_area ? &start_b : &end_b };
			const DVec3 &sample_normal = start_has_area ? start_source_normal : end_source_normal;
			const uint64_t emit_keys[3] = { key_a, key_c, key_b };
			const uint32_t transition_flags = static_cast<uint32_t>(scratch.tri_transition_on[t]) |
					(static_cast<uint32_t>(scratch.tri_transition_off[t]) << 6);
			for (int corner = 0; corner < 3; ++corner) {
				const DVec3 p = (*emit_start[corner]) * v_scale + v_offset;
				vertices_w[vertex_count] = Vector3(
						static_cast<real_t>(p.x), static_cast<real_t>(p.y), static_cast<real_t>(p.z));
				normals_w[vertex_count] = gd_normal;
				colors_w[vertex_count] = params.default_color;
				uvs_w[vertex_count] = triplanar_uv(*sample_positions[corner], sample_normal);
				if (lod_attribs_w != nullptr) {
					const size_t li = vertex_count * 4;
					if (timeline_w != nullptr) {
						// With a timeline present, CUSTOM0.xyz carries the
						// interval-END face normal (the shader lerps NORMAL
						// toward it); the old position copy carried no
						// information the shaders used.
						lod_attribs_w[li] = static_cast<float>(end_normal.x);
						lod_attribs_w[li + 1] = static_cast<float>(end_normal.y);
						lod_attribs_w[li + 2] = static_cast<float>(end_normal.z);
					} else {
						lod_attribs_w[li] = static_cast<float>(p.x);
						lod_attribs_w[li + 1] = static_cast<float>(p.y);
						lod_attribs_w[li + 2] = static_cast<float>(p.z);
					}
					memcpy(&lod_attribs_w[li + 3], &transition_flags, sizeof(float));
				}
				if (timeline_w != nullptr) {
					const size_t wi = vertex_count * 4;
					const DVec3 ep = (*emit_end[corner]) * v_scale + v_offset;
					timeline_w[wi] = static_cast<float>(ep.x);
					timeline_w[wi + 1] = static_cast<float>(ep.y);
					timeline_w[wi + 2] = static_cast<float>(ep.z);
					timeline_w[wi + 3] = packed_window;
				}
				if (tex_attribs_w != nullptr) {
					const size_t xi = vertex_count * 2;
					sample_tex_attrib(
							*sample_positions[corner], sample_normal, tex_attribs_w[xi], tex_attribs_w[xi + 1]);
				}
				if (tangents_w != nullptr) {
					const size_t ti = vertex_count * 4;
					tangents_w[ti] = static_cast<float>(tangent.x);
					tangents_w[ti + 1] = static_cast<float>(tangent.y);
					tangents_w[ti + 2] = static_cast<float>(tangent.z);
					tangents_w[ti + 3] = tangent_w;
				}
				if (input.collision_hint && scratch.tri_transition_on[t] == 0 && scratch.tri_t_in[t] == 0.f &&
						start_has_area) {
					auto found = collision_vertex_indices.find(emit_keys[corner]);
					int collision_index;
					if (found == collision_vertex_indices.end()) {
						collision_index = output.collision_surface.positions.size();
						collision_vertex_indices.insert({ emit_keys[corner], collision_index });
						output.collision_surface.positions.push_back(Vector3f(p.x, p.y, p.z));
					} else {
						collision_index = found->second;
					}
					output.collision_surface.indices.push_back(collision_index);
				}
				++vertex_count;
			}
		}

		if (vertex_count == 0) {
			TETRA_DBG("RET no vertices");
			return;
		}
		TETRA_DBG("emit verts=%d", int(vertex_count));
		out_vertices.resize(vertex_count);
		out_normals.resize(vertex_count);
		out_colors.resize(vertex_count);
		out_uvs.resize(vertex_count);
		if (tangents_w != nullptr) {
			out_tangents.resize(vertex_count * 4);
		}
		if (lod_attribs_w != nullptr) {
			out_lod_attribs.resize(vertex_count * 4);
		}
		if (tex_attribs_w != nullptr) {
			out_tex_attribs.resize(vertex_count * 2);
		}
		if (timeline_w != nullptr) {
			out_timeline_attribs.resize(vertex_count * 4);
		}
	}

	// Sequential indices: the vertices are already split per triangle for flat
	// shading, but godot_voxel's is_surface_triangulated() helper (used by the
	// script API and by mesh_block_task) requires an index array, and indexed
	// surfaces keep collision baking on the standard path too.
	PackedInt32Array out_indices;
	{
		const int64_t vertex_count = out_vertices.size();
		out_indices.resize(vertex_count);
		int32_t *w = out_indices.ptrw();
		for (int64_t i = 0; i < vertex_count; ++i) {
			w[i] = static_cast<int32_t>(i);
		}
	}

	Output::Surface surface;
	Array &mesh_arrays = surface.arrays;
	mesh_arrays.resize(Mesh::ARRAY_MAX);
	mesh_arrays[Mesh::ARRAY_VERTEX] = out_vertices;
	mesh_arrays[Mesh::ARRAY_NORMAL] = out_normals;
	mesh_arrays[Mesh::ARRAY_COLOR] = out_colors;
	mesh_arrays[Mesh::ARRAY_TEX_UV] = out_uvs;
	if (params.generate_tangents) {
		mesh_arrays[Mesh::ARRAY_TANGENT] = out_tangents;
	}
	if (input.lod_hint) {
		mesh_arrays[Mesh::ARRAY_CUSTOM0] = out_lod_attribs;
		output.mesh_flags |= (RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM0_SHIFT);
	}
	if (params.texturing_enabled) {
		// Two floats per vertex: x = 4 packed indices, y = 4 packed weights.
		// The format flag is not optional -- without it the RenderingServer
		// reads CUSTOM1 as the default (8-bit unorm) and every index comes
		// back as a fraction.
		mesh_arrays[Mesh::ARRAY_CUSTOM1] = out_tex_attribs;
		output.mesh_flags |= (RenderingServerEnums::ARRAY_CUSTOM_RG_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM1_SHIFT);
	}
	if (!out_timeline_attribs.is_empty()) {
		// The geomorph timeline rides in CUSTOM2 (see out_timeline_attribs);
		// the shader collapses each slot outside its window and lerps inside
		// it. No blend shapes: the whole morph is evaluated in the material's
		// vertex stage.
		mesh_arrays[Mesh::ARRAY_CUSTOM2] = out_timeline_attribs;
		output.mesh_flags |= (RenderingServerEnums::ARRAY_CUSTOM_RGBA_FLOAT << Mesh::ARRAY_FORMAT_CUSTOM2_SHIFT);
	}
	mesh_arrays[Mesh::ARRAY_INDEX] = out_indices;
	output.surfaces.push_back(surface);
	output.primitive_type = Mesh::PRIMITIVE_TRIANGLES;
	TETRA_DBG("surface pushed");
}

// ---------------------------------------------------------------------------

void VoxelMesherTetra::set_subdivisions(int p_subdivisions) {
	_parameters.subdivisions = CLAMP(p_subdivisions, MIN_SUBDIVISIONS, MAX_SUBDIVISIONS);
	emit_changed();
}

int VoxelMesherTetra::get_subdivisions() const {
	return _parameters.subdivisions;
}

void VoxelMesherTetra::set_warp_enabled(bool p_enabled) {
	_parameters.warp_enabled = p_enabled;
	emit_changed();
}

bool VoxelMesherTetra::get_warp_enabled() const {
	return _parameters.warp_enabled;
}

void VoxelMesherTetra::set_lattice_warp(float p_amount) {
	_parameters.lattice_warp = CLAMP(p_amount, 0.f, 0.75f);
	emit_changed();
}

float VoxelMesherTetra::get_lattice_warp() const {
	return _parameters.lattice_warp;
}

void VoxelMesherTetra::set_patch_relief(float p_amount) {
	_parameters.patch_relief = CLAMP(p_amount, 0.f, 0.45f);
	emit_changed();
}

float VoxelMesherTetra::get_patch_relief() const {
	return _parameters.patch_relief;
}

void VoxelMesherTetra::set_patch_size(float p_size) {
	_parameters.patch_size = CLAMP(p_size, 0.05f, 3.f);
	emit_changed();
}

float VoxelMesherTetra::get_patch_size() const {
	return _parameters.patch_size;
}

void VoxelMesherTetra::set_patch_relief_smoothing(float p_smoothing) {
	_parameters.patch_relief_smoothing = CLAMP(p_smoothing, 0.f, 1.f);
	emit_changed();
}

float VoxelMesherTetra::get_patch_relief_smoothing() const {
	return _parameters.patch_relief_smoothing;
}

void VoxelMesherTetra::set_default_color(Color p_color) {
	_parameters.default_color = p_color;
	emit_changed();
}

Color VoxelMesherTetra::get_default_color() const {
	return _parameters.default_color;
}

void VoxelMesherTetra::set_texturing_enabled(bool p_enabled) {
	_parameters.texturing_enabled = p_enabled;
	emit_changed();
}

bool VoxelMesherTetra::get_texturing_enabled() const {
	return _parameters.texturing_enabled;
}

void VoxelMesherTetra::set_generate_tangents(bool p_enabled) {
	_parameters.generate_tangents = p_enabled;
	emit_changed();
}

bool VoxelMesherTetra::get_generate_tangents() const {
	return _parameters.generate_tangents;
}

void VoxelMesherTetra::set_lod_transitions_enabled(bool p_enabled) {
	_parameters.lod_transitions_enabled = p_enabled;
	emit_changed();
}

bool VoxelMesherTetra::get_lod_transitions_enabled() const {
	return _parameters.lod_transitions_enabled;
}

void VoxelMesherTetra::set_lod_geomorph_enabled(bool p_enabled) {
	_parameters.lod_geomorph_enabled = p_enabled;
	emit_changed();
}

bool VoxelMesherTetra::get_lod_geomorph_enabled() const {
	return _parameters.lod_geomorph_enabled;
}

void VoxelMesherTetra::set_vertex_scale(float p_scale) {
	_parameters.vertex_scale = MAX(0.0001f, p_scale);
	emit_changed();
}

float VoxelMesherTetra::get_vertex_scale() const {
	return _parameters.vertex_scale;
}

void VoxelMesherTetra::set_vertex_offset(Vector3 p_offset) {
	_parameters.vertex_offset = p_offset;
	emit_changed();
}

Vector3 VoxelMesherTetra::stylize_world_point(
		Vector3 p_world_position, Vector3 p_world_normal, float p_voxel_size) const {
	if (!_parameters.warp_enabled || p_voxel_size <= 0.f) {
		return p_world_position;
	}
	const double vs = static_cast<double>(p_voxel_size);
	const DVec3 p(static_cast<double>(p_world_position.x) / vs, static_cast<double>(p_world_position.y) / vs,
			static_cast<double>(p_world_position.z) / vs);

	// The relief term is the only consumer of the normal, and it is skipped
	// entirely when patch_relief is 0 -- but a caller that passes a degenerate
	// normal should get the warp rather than a NaN.
	DVec3 n(static_cast<double>(p_world_normal.x), static_cast<double>(p_world_normal.y),
			static_cast<double>(p_world_normal.z));
	if (n.length_squared() < 1.0e-12) {
		n = DVec3(0.0, 1.0, 0.0);
	} else {
		n = n.normalized();
	}

	const DVec3 w = stylized_surface_position(p, n, _parameters.lattice_warp, _parameters.patch_relief,
			_parameters.patch_size, _parameters.patch_relief_smoothing);
	return Vector3(static_cast<real_t>(w.x * vs), static_cast<real_t>(w.y * vs), static_cast<real_t>(w.z * vs));
}

Vector3 VoxelMesherTetra::get_vertex_offset() const {
	return _parameters.vertex_offset;
}

void VoxelMesherTetra::set_collision_subdivisions(int p_subdivisions) {
	_parameters.collision_subdivisions = CLAMP(p_subdivisions, MIN_SUBDIVISIONS, MAX_SUBDIVISIONS);
	emit_changed();
}

int VoxelMesherTetra::get_collision_subdivisions() const {
	return _parameters.collision_subdivisions;
}

void VoxelMesherTetra::set_uv_scale(float p_scale) {
	_parameters.uv_scale = CLAMP(p_scale, 0.f, 16.f);
	emit_changed();
}

float VoxelMesherTetra::get_uv_scale() const {
	return _parameters.uv_scale;
}

void VoxelMesherTetra::_bind_methods() {
	ClassDB::bind_method(D_METHOD("stylize_world_point", "world_position", "world_normal", "voxel_size"),
			&VoxelMesherTetra::stylize_world_point);

	ClassDB::bind_method(D_METHOD("set_subdivisions", "subdivisions"), &VoxelMesherTetra::set_subdivisions);
	ClassDB::bind_method(D_METHOD("get_subdivisions"), &VoxelMesherTetra::get_subdivisions);

	ClassDB::bind_method(D_METHOD("set_warp_enabled", "enabled"), &VoxelMesherTetra::set_warp_enabled);
	ClassDB::bind_method(D_METHOD("get_warp_enabled"), &VoxelMesherTetra::get_warp_enabled);

	ClassDB::bind_method(D_METHOD("set_lattice_warp", "amount"), &VoxelMesherTetra::set_lattice_warp);
	ClassDB::bind_method(D_METHOD("get_lattice_warp"), &VoxelMesherTetra::get_lattice_warp);

	ClassDB::bind_method(D_METHOD("set_patch_relief", "amount"), &VoxelMesherTetra::set_patch_relief);
	ClassDB::bind_method(D_METHOD("get_patch_relief"), &VoxelMesherTetra::get_patch_relief);

	ClassDB::bind_method(D_METHOD("set_patch_size", "size"), &VoxelMesherTetra::set_patch_size);
	ClassDB::bind_method(D_METHOD("get_patch_size"), &VoxelMesherTetra::get_patch_size);

	ClassDB::bind_method(
			D_METHOD("set_patch_relief_smoothing", "smoothing"), &VoxelMesherTetra::set_patch_relief_smoothing);
	ClassDB::bind_method(D_METHOD("get_patch_relief_smoothing"), &VoxelMesherTetra::get_patch_relief_smoothing);

	ClassDB::bind_method(D_METHOD("set_default_color", "color"), &VoxelMesherTetra::set_default_color);
	ClassDB::bind_method(D_METHOD("get_default_color"), &VoxelMesherTetra::get_default_color);

	ClassDB::bind_method(D_METHOD("set_texturing_enabled", "enabled"), &VoxelMesherTetra::set_texturing_enabled);
	ClassDB::bind_method(D_METHOD("get_texturing_enabled"), &VoxelMesherTetra::get_texturing_enabled);

	ClassDB::bind_method(D_METHOD("set_generate_tangents", "enabled"), &VoxelMesherTetra::set_generate_tangents);
	ClassDB::bind_method(D_METHOD("get_generate_tangents"), &VoxelMesherTetra::get_generate_tangents);
	ClassDB::bind_method(
			D_METHOD("set_lod_transitions_enabled", "enabled"), &VoxelMesherTetra::set_lod_transitions_enabled);
	ClassDB::bind_method(D_METHOD("get_lod_transitions_enabled"), &VoxelMesherTetra::get_lod_transitions_enabled);
	ClassDB::bind_method(D_METHOD("set_lod_geomorph_enabled", "enabled"), &VoxelMesherTetra::set_lod_geomorph_enabled);
	ClassDB::bind_method(D_METHOD("get_lod_geomorph_enabled"), &VoxelMesherTetra::get_lod_geomorph_enabled);

	ClassDB::bind_method(D_METHOD("set_vertex_scale", "scale"), &VoxelMesherTetra::set_vertex_scale);
	ClassDB::bind_method(D_METHOD("get_vertex_scale"), &VoxelMesherTetra::get_vertex_scale);

	ClassDB::bind_method(D_METHOD("set_vertex_offset", "offset"), &VoxelMesherTetra::set_vertex_offset);
	ClassDB::bind_method(D_METHOD("get_vertex_offset"), &VoxelMesherTetra::get_vertex_offset);

	ClassDB::bind_method(
			D_METHOD("set_collision_subdivisions", "subdivisions"), &VoxelMesherTetra::set_collision_subdivisions);
	ClassDB::bind_method(D_METHOD("get_collision_subdivisions"), &VoxelMesherTetra::get_collision_subdivisions);

	ClassDB::bind_method(D_METHOD("set_uv_scale", "scale"), &VoxelMesherTetra::set_uv_scale);
	ClassDB::bind_method(D_METHOD("get_uv_scale"), &VoxelMesherTetra::get_uv_scale);

	ADD_PROPERTY(
			PropertyInfo(Variant::INT, "subdivisions", PROPERTY_HINT_RANGE, "1,4,1"),
			"set_subdivisions",
			"get_subdivisions");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "warp_enabled"), "set_warp_enabled", "get_warp_enabled");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "lattice_warp", PROPERTY_HINT_RANGE, "0.0,0.75,0.01"),
			"set_lattice_warp",
			"get_lattice_warp");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "patch_relief", PROPERTY_HINT_RANGE, "0.0,0.45,0.01"),
			"set_patch_relief",
			"get_patch_relief");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "patch_size", PROPERTY_HINT_RANGE, "0.05,3.0,0.05"),
			"set_patch_size",
			"get_patch_size");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "patch_relief_smoothing", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"),
			"set_patch_relief_smoothing",
			"get_patch_relief_smoothing");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "default_color"), "set_default_color", "get_default_color");
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "texturing_enabled"), "set_texturing_enabled", "get_texturing_enabled");
	ADD_PROPERTY(
			PropertyInfo(Variant::BOOL, "generate_tangents"), "set_generate_tangents", "get_generate_tangents");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lod_transitions_enabled"), "set_lod_transitions_enabled", "get_lod_transitions_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "lod_geomorph_enabled"), "set_lod_geomorph_enabled", "get_lod_geomorph_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "vertex_scale"), "set_vertex_scale", "get_vertex_scale");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "vertex_offset"), "set_vertex_offset", "get_vertex_offset");
	ADD_GROUP("Collision", "collision_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_subdivisions", PROPERTY_HINT_RANGE, "1,4,1"),
			"set_collision_subdivisions", "get_collision_subdivisions");
	ADD_GROUP("", "");
	ADD_PROPERTY(
			PropertyInfo(Variant::FLOAT, "uv_scale", PROPERTY_HINT_RANGE, "0.0,16.0,0.01"),
			"set_uv_scale",
			"get_uv_scale");
}

} // namespace zylann::voxel
