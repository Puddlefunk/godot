#include "test_voxel_mesher_tetra.h"
#include "../../meshers/tetra/voxel_mesher_tetra.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/testing/test_macros.h"

#include <algorithm>
#include <cstring>

namespace zylann::voxel::tests {

namespace {

// 2 min padding + 4 voxels of content + 3 max padding -- must track
// VoxelMesherTetra's set_padding(2, 3) (the extra max layer keeps the
// parent lattice covering the accumulation skirt).
constexpr int TEST_BUFFER_SIZE = 9;

VoxelMesher::Output build_plane(
		const Ref<VoxelMesherTetra> &mesher, Vector3i content_origin, uint8_t lod_index, float plane_height,
		bool collision_hint = false) {
	const int lod_stride = 1 << lod_index;
	const Vector3i padded_origin = content_origin - Vector3iUtil::create(2 * lod_stride);

	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_32_BIT);
	voxels.create(Vector3iUtil::create(TEST_BUFFER_SIZE));
	for (int z = 0; z < TEST_BUFFER_SIZE; ++z) {
		for (int y = 0; y < TEST_BUFFER_SIZE; ++y) {
			for (int x = 0; x < TEST_BUFFER_SIZE; ++x) {
				const float world_y = padded_origin.y + y * lod_stride;
				voxels.set_voxel_f((world_y - plane_height) / lod_stride, x, y, z, VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
	VoxelMesher::Output output;
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, content_origin, lod_index, collision_hint, true, false });
	return output;
}

VoxelMesher::Output build_height_field(
		const Ref<VoxelMesherTetra> &mesher, Vector3i content_origin, uint8_t lod_index) {
	const int lod_stride = 1 << lod_index;
	const Vector3i padded_origin = content_origin - Vector3iUtil::create(2 * lod_stride);

	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_32_BIT);
	voxels.create(Vector3iUtil::create(TEST_BUFFER_SIZE));
	for (int z = 0; z < TEST_BUFFER_SIZE; ++z) {
		for (int y = 0; y < TEST_BUFFER_SIZE; ++y) {
			for (int x = 0; x < TEST_BUFFER_SIZE; ++x) {
				const Vector3 world_position = Vector3(padded_origin) + Vector3(x, y, z) * lod_stride;
				const float height = 1.25f + 0.08f * world_position.z * world_position.z +
						0.13f * Math::sin(world_position.x * 0.7f + world_position.z * 0.4f);
				voxels.set_voxel_f(
						(world_position.y - height) / lod_stride, x, y, z, VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
	VoxelMesher::Output output;
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, content_origin, lod_index, false, true, false });
	return output;
}

struct Segment {
	Vector3 a;
	Vector3 b;
};

struct Triangle {
	Vector3 a;
	Vector3 b;
	Vector3 c;
};

double sample_kuhn_cube(const double values[8], const double f[3]) {
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
		int coordinates[3] = { 0, 0, 0 };
		if (a >= 0) {
			coordinates[a] = 1;
		}
		if (b >= 0) {
			coordinates[b] = 1;
		}
		static const int8_t corners[8][3] = {
			{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 },
			{ 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }
		};
		for (int i = 0; i < 8; ++i) {
			if (corners[i][0] == coordinates[0] && corners[i][1] == coordinates[1] &&
					corners[i][2] == coordinates[2]) {
				return i;
			}
		}
		return 0;
	};
	const int c1 = corner_from_axes(axes[0], -1);
	const int c2 = corner_from_axes(axes[0], axes[1]);
	return values[0] * (1.0 - f[axes[0]]) + values[c1] * (f[axes[0]] - f[axes[1]]) +
			values[c2] * (f[axes[1]] - f[axes[2]]) + values[6] * f[axes[2]];
}

VoxelMesher::Output build_parent_p1_field(
		const Ref<VoxelMesherTetra> &mesher, Vector3i content_origin, uint8_t lod_index) {
	static const int8_t corners[8][3] = {
		{ 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 },
		{ 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }
	};
	const int lod_stride = 1 << lod_index;
	const int parent_stride = 2;
	const Vector3i padded_origin = content_origin - Vector3iUtil::create(2 * lod_stride);
	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_32_BIT);
	voxels.create(Vector3iUtil::create(TEST_BUFFER_SIZE));
	for (int z = 0; z < TEST_BUFFER_SIZE; ++z) {
		for (int y = 0; y < TEST_BUFFER_SIZE; ++y) {
			for (int x = 0; x < TEST_BUFFER_SIZE; ++x) {
				const Vector3i p = padded_origin + Vector3i(x, y, z) * lod_stride;
				const Vector3i base(
						static_cast<int>(Math::floor(static_cast<double>(p.x) / parent_stride)) * parent_stride,
						static_cast<int>(Math::floor(static_cast<double>(p.y) / parent_stride)) * parent_stride,
						static_cast<int>(Math::floor(static_cast<double>(p.z) / parent_stride)) * parent_stride);
				const double f[3] = {
					static_cast<double>(p.x - base.x) / parent_stride,
					static_cast<double>(p.y - base.y) / parent_stride,
					static_cast<double>(p.z - base.z) / parent_stride
				};
				double values[8];
				for (int ci = 0; ci < 8; ++ci) {
					const Vector3i q = base + Vector3i(corners[ci][0], corners[ci][1], corners[ci][2]) * parent_stride;
					const double height = 3.2 + 0.12 * q.x + 0.08 * q.z + 0.65 * Math::sin(q.x * 0.37 + q.z * 0.21);
					values[ci] = height - q.y;
				}
				const double density = sample_kuhn_cube(values, f);
				voxels.set_voxel_f(static_cast<float>(-density / lod_stride), x, y, z, VoxelBuffer::CHANNEL_SDF);
			}
		}
	}
	VoxelMesher::Output output;
	mesher->build(output, VoxelMesher::Input{ voxels, nullptr, content_origin, lod_index, false, true, false });
	return output;
}

void append_triangles(
		const VoxelMesher::Output &output, Vector3i origin, bool use_morph_target, StdVector<Triangle> &triangles) {
	if (output.surfaces.empty()) {
		return;
	}
	const Array arrays = output.surfaces[0].arrays;
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	PackedFloat32Array timeline;
	if (use_morph_target) {
		// Geomorph builds carry their timeline in CUSTOM2 (blend shapes are
		// gone): xyz = interval-end position, w = packed [t_in, t_out]. The
		// w=1 surface is the set of slots whose window ends at t_out code
		// 4095, read at their end positions -- births included.
		ZN_TEST_ASSERT(output.surfaces[0].blend_shapes.size() == 0);
		timeline = arrays[Mesh::ARRAY_CUSTOM2];
		ZN_TEST_ASSERT(timeline.size() == vertices.size() * 4);
	}
	for (int i = 0; i < vertices.size(); i += 3) {
		Triangle triangle;
		if (use_morph_target) {
			const uint32_t packed = static_cast<uint32_t>(timeline[i * 4 + 3]);
			if ((packed % 4096u) != 4095u) {
				continue;
			}
			triangle.a = Vector3(timeline[i * 4], timeline[i * 4 + 1], timeline[i * 4 + 2]) + Vector3(origin);
			triangle.b = Vector3(timeline[(i + 1) * 4], timeline[(i + 1) * 4 + 1], timeline[(i + 1) * 4 + 2]) +
					Vector3(origin);
			triangle.c = Vector3(timeline[(i + 2) * 4], timeline[(i + 2) * 4 + 1], timeline[(i + 2) * 4 + 2]) +
					Vector3(origin);
		} else {
			triangle = Triangle{ vertices[i] + Vector3(origin), vertices[i + 1] + Vector3(origin),
				vertices[i + 2] + Vector3(origin) };
		}
		if ((triangle.b - triangle.a).cross(triangle.c - triangle.a).length_squared() > 1.0e-10f) {
			triangles.push_back(triangle);
		}
	}
}

float point_triangle_distance_squared(const Vector3 &p, const Triangle &triangle) {
	const Vector3 ab = triangle.b - triangle.a;
	const Vector3 ac = triangle.c - triangle.a;
	const Vector3 ap = p - triangle.a;
	const float d1 = ab.dot(ap);
	const float d2 = ac.dot(ap);
	if (d1 <= 0.f && d2 <= 0.f) {
		return p.distance_squared_to(triangle.a);
	}
	const Vector3 bp = p - triangle.b;
	const float d3 = ab.dot(bp);
	const float d4 = ac.dot(bp);
	if (d3 >= 0.f && d4 <= d3) {
		return p.distance_squared_to(triangle.b);
	}
	const float vc = d1 * d4 - d3 * d2;
	if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
		return p.distance_squared_to(triangle.a + ab * (d1 / (d1 - d3)));
	}
	const Vector3 cp = p - triangle.c;
	const float d5 = ab.dot(cp);
	const float d6 = ac.dot(cp);
	if (d6 >= 0.f && d5 <= d6) {
		return p.distance_squared_to(triangle.c);
	}
	const float vb = d5 * d2 - d1 * d6;
	if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
		return p.distance_squared_to(triangle.a + ac * (d2 / (d2 - d6)));
	}
	const float va = d3 * d6 - d5 * d4;
	if (va <= 0.f && d4 >= d3 && d5 >= d6) {
		const Vector3 bc = triangle.c - triangle.b;
		return p.distance_squared_to(triangle.b + bc * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
	}
	const float denominator = 1.f / (va + vb + vc);
	const Vector3 closest = triangle.a + ab * (vb * denominator) + ac * (vc * denominator);
	return p.distance_squared_to(closest);
}

void assert_points_on_surface(const StdVector<Triangle> &source, const StdVector<Triangle> &surface) {
	for (const Triangle &triangle : source) {
		const Vector3 points[4] = { triangle.a, triangle.b, triangle.c, (triangle.a + triangle.b + triangle.c) / 3.f };
		for (const Vector3 &point : points) {
			float minimum_distance_squared = 1.0e10f;
			for (const Triangle &surface_triangle : surface) {
				minimum_distance_squared = MIN(
						minimum_distance_squared, point_triangle_distance_squared(point, surface_triangle));
			}
			ZN_TEST_ASSERT(minimum_distance_squared < 1.0e-7f);
		}
	}
}

StdVector<Segment> get_visible_border_segments(
		const VoxelMesher::Output &output, Vector3i origin, float local_x, uint32_t transition_mask) {
	StdVector<Segment> segments;
	const Array arrays = output.surfaces[0].arrays;
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const PackedFloat32Array lod_data = arrays[Mesh::ARRAY_CUSTOM0];
	for (int triangle = 0; triangle < vertices.size(); triangle += 3) {
		uint32_t flags = 0;
		memcpy(&flags, &lod_data[triangle * 4 + 3], sizeof(uint32_t));
		const uint32_t transition_on = flags & 63;
		const uint32_t transition_off = (flags >> 6) & 63;
		if ((transition_mask & transition_on) != transition_on || (transition_mask & transition_off) != 0) {
			continue;
		}
		Vector3 border[3];
		int border_count = 0;
		for (int corner = 0; corner < 3; ++corner) {
			const Vector3 vertex = vertices[triangle + corner];
			if (Math::is_equal_approx(vertex.x, local_x)) {
				border[border_count++] = vertex + Vector3(origin);
			}
		}
		if (border_count >= 2 && !border[0].is_equal_approx(border[1])) {
			segments.push_back(Segment{ border[0], border[1] });
		}
	}
	return segments;
}

float point_segment_distance_squared(Vector3 point, const Segment &segment) {
	const Vector3 delta = segment.b - segment.a;
	const float length_squared = delta.length_squared();
	if (length_squared == 0.f) {
		return point.distance_squared_to(segment.a);
	}
	const float t = CLAMP((point - segment.a).dot(delta) / length_squared, 0.f, 1.f);
	return point.distance_squared_to(segment.a + delta * t);
}

StdVector<Vector3> get_collision_border_vertices(
		const VoxelMesher::Output &output, Vector3i origin, float local_x) {
	StdVector<Vector3> border;
	for (const int index : output.collision_surface.indices) {
		ZN_TEST_ASSERT(index >= 0 && static_cast<size_t>(index) < output.collision_surface.positions.size());
		const Vector3f vertex = output.collision_surface.positions[index];
		if (!Math::is_equal_approx(vertex.x, local_x)) {
			continue;
		}
		const Vector3 world_vertex = Vector3(vertex.x, vertex.y, vertex.z) + Vector3(origin);
		bool found = false;
		for (const Vector3 existing : border) {
			if (existing.is_equal_approx(world_vertex)) {
				found = true;
				break;
			}
		}
		if (!found) {
			border.push_back(world_vertex);
		}
	}
	return border;
}

StdVector<Vector3> get_border_vertices(const VoxelMesher::Output &output, Vector3i origin, float local_x) {
	StdVector<Vector3> border;
	const PackedVector3Array vertices = output.surfaces[0].arrays[Mesh::ARRAY_VERTEX];
	for (const Vector3 vertex : vertices) {
		if (!Math::is_equal_approx(vertex.x, local_x)) {
			continue;
		}
		const Vector3 world_vertex = vertex + Vector3(origin);
		bool found = false;
		for (const Vector3 existing : border) {
			if (existing.is_equal_approx(world_vertex)) {
				found = true;
				break;
			}
		}
		if (!found) {
			border.push_back(world_vertex);
		}
	}
	return border;
}

} // namespace

void test_voxel_mesher_tetra_lod_coordinates() {
	const uint8_t lod_index = 2;
	const int lod_stride = 1 << lod_index;
	const Vector3i content_origin(16, 8, 24);
	const float plane_height = 13.f;

	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(1);
	mesher->set_warp_enabled(false);
	ZN_TEST_ASSERT(mesher->supports_lod());

	const VoxelMesher::Output output = build_plane(mesher, content_origin, lod_index, plane_height);
	ZN_TEST_ASSERT(output.surfaces.size() == 1);

	const PackedVector3Array vertices = output.surfaces[0].arrays[Mesh::ARRAY_VERTEX];
	ZN_TEST_ASSERT(vertices.size() > 0);
	const float expected_local_y = plane_height - content_origin.y;
	const float content_extent = 4 * lod_stride;
	for (const Vector3 vertex : vertices) {
		ZN_TEST_ASSERT(Math::is_equal_approx(vertex.y, expected_local_y));
		ZN_TEST_ASSERT(vertex.x >= 0.f && vertex.x <= content_extent);
		ZN_TEST_ASSERT(vertex.z >= 0.f && vertex.z <= content_extent);
	}
}

void test_voxel_mesher_tetra_same_lod_border() {
	const uint8_t lod_index = 2;
	const int content_extent = 4 * (1 << lod_index);
	const Vector3i left_origin(16, 8, 24);
	const Vector3i right_origin = left_origin + Vector3i(content_extent, 0, 0);

	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(1);
	mesher->set_lattice_warp(0.f);
	mesher->set_patch_relief(0.2f);
	mesher->set_patch_size(6.5f);
	mesher->set_patch_relief_smoothing(0.5f);

	const VoxelMesher::Output left = build_plane(mesher, left_origin, lod_index, 13.f);
	const VoxelMesher::Output right = build_plane(mesher, right_origin, lod_index, 13.f);
	ZN_TEST_ASSERT(left.surfaces.size() == 1);
	ZN_TEST_ASSERT(right.surfaces.size() == 1);

	const StdVector<Vector3> left_border = get_border_vertices(left, left_origin, content_extent);
	const StdVector<Vector3> right_border = get_border_vertices(right, right_origin, 0.f);
	ZN_TEST_ASSERT(left_border.size() > 0);
	ZN_TEST_ASSERT(left_border.size() == right_border.size());
	for (const Vector3 left_vertex : left_border) {
		bool found = false;
		for (const Vector3 right_vertex : right_border) {
			if (left_vertex.is_equal_approx(right_vertex)) {
				found = true;
				break;
			}
		}
		ZN_TEST_ASSERT(found);
	}
}

void test_voxel_mesher_tetra_collision_surface() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_lattice_warp(0.f);
	mesher->set_patch_relief(0.2f);
	mesher->set_patch_size(6.5f);
	mesher->set_patch_relief_smoothing(0.5f);
	mesher->set_collision_subdivisions(1);
	ZN_TEST_ASSERT(mesher->is_generating_collision_surface());

	const VoxelMesher::Output output = build_plane(mesher, Vector3i(0, 0, 0), 0, 1.3f, true);
	ZN_TEST_ASSERT(output.surfaces.size() == 1);
	ZN_TEST_ASSERT(output.collision_surface.positions.size() > 0);
	ZN_TEST_ASSERT(output.collision_surface.indices.size() >= 3);
	ZN_TEST_ASSERT(output.collision_surface.indices.size() % 3 == 0);
	const PackedInt32Array render_indices = output.surfaces[0].arrays[Mesh::ARRAY_INDEX];
	ZN_TEST_ASSERT(output.collision_surface.indices.size() < static_cast<size_t>(render_indices.size()));
	for (const int index : output.collision_surface.indices) {
		ZN_TEST_ASSERT(index >= 0 && static_cast<size_t>(index) < output.collision_surface.positions.size());
	}
}

void test_voxel_mesher_tetra_collision_border() {
	const int content_extent = 4;
	const Vector3i left_origin(0, 0, 0);
	const Vector3i right_origin(content_extent, 0, 0);

	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_lattice_warp(0.f);
	mesher->set_patch_relief(0.2f);
	mesher->set_patch_size(6.5f);
	mesher->set_patch_relief_smoothing(0.5f);
	mesher->set_collision_subdivisions(1);

	const VoxelMesher::Output left = build_plane(mesher, left_origin, 0, 1.3f, true);
	const VoxelMesher::Output right = build_plane(mesher, right_origin, 0, 1.3f, true);
	const StdVector<Vector3> left_border = get_collision_border_vertices(left, left_origin, content_extent);
	const StdVector<Vector3> right_border = get_collision_border_vertices(right, right_origin, 0.f);
	ZN_TEST_ASSERT(left_border.size() > 0);
	ZN_TEST_ASSERT(left_border.size() == right_border.size());
	for (const Vector3 left_vertex : left_border) {
		bool found = false;
		for (const Vector3 right_vertex : right_border) {
			if (left_vertex.is_equal_approx(right_vertex)) {
				found = true;
				break;
			}
		}
		ZN_TEST_ASSERT(found);
	}
}

void test_voxel_mesher_tetra_transition_variants() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_warp_enabled(false);

	const VoxelMesher::Output output = build_plane(mesher, Vector3i(0, 0, 0), 0, 1.3f);
	ZN_TEST_ASSERT(output.surfaces.size() == 1);
	const Array arrays = output.surfaces[0].arrays;
	const PackedVector3Array vertices = arrays[Mesh::ARRAY_VERTEX];
	const PackedFloat32Array lod_data = arrays[Mesh::ARRAY_CUSTOM0];
	ZN_TEST_ASSERT(vertices.size() > 0);
	ZN_TEST_ASSERT(vertices.size() % 3 == 0);
	ZN_TEST_ASSERT(lod_data.size() == vertices.size() * 4);

	bool found_transition_on = false;
	bool found_transition_off = false;
	for (int vertex_index = 0; vertex_index < vertices.size(); vertex_index += 3) {
		uint32_t triangle_flags = 0;
		memcpy(&triangle_flags, &lod_data[vertex_index * 4 + 3], sizeof(uint32_t));
		for (int corner = 0; corner < 3; ++corner) {
			const int i = vertex_index + corner;
			uint32_t flags = 0;
			memcpy(&flags, &lod_data[i * 4 + 3], sizeof(uint32_t));
			ZN_TEST_ASSERT(flags == triangle_flags);
			ZN_TEST_ASSERT((flags >> 12) == 0);
			const uint32_t transition_on = flags & 63;
			const uint32_t transition_off = (flags >> 6) & 63;
			ZN_TEST_ASSERT((transition_on & transition_off) == 0);
			found_transition_on |= transition_on != 0;
			found_transition_off |= transition_off != 0;
			const Vector3 secondary(lod_data[i * 4], lod_data[i * 4 + 1], lod_data[i * 4 + 2]);
			ZN_TEST_ASSERT(secondary.is_equal_approx(vertices[i]));
		}
	}
	ZN_TEST_ASSERT(found_transition_on);
	ZN_TEST_ASSERT(found_transition_off);
}

void test_voxel_mesher_tetra_mixed_lod_border() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_warp_enabled(false);

	const Vector3i fine_origin(4, 0, 0);
	const Vector3i coarse_origin(8, 0, 0);
	const VoxelMesher::Output fine = build_height_field(mesher, fine_origin, 0);
	const VoxelMesher::Output coarse = build_height_field(mesher, coarse_origin, 1);
	ZN_TEST_ASSERT(fine.surfaces.size() == 1);
	ZN_TEST_ASSERT(coarse.surfaces.size() == 1);

	const StdVector<Segment> fine_segments = get_visible_border_segments(fine, fine_origin, 4.f, 2);
	const StdVector<Segment> coarse_segments = get_visible_border_segments(coarse, coarse_origin, 0.f, 0);
	ZN_TEST_ASSERT(fine_segments.size() > 0);
	ZN_TEST_ASSERT(coarse_segments.size() > 0);
	for (const Segment &fine_segment : fine_segments) {
		for (const Vector3 point : { fine_segment.a, fine_segment.b }) {
			float minimum_distance_squared = 1.0e10f;
			for (const Segment &coarse_segment : coarse_segments) {
				minimum_distance_squared = MIN(
						minimum_distance_squared, point_segment_distance_squared(point, coarse_segment));
			}
			ZN_TEST_ASSERT(minimum_distance_squared < 1.0e-8f);
		}
	}
}

void test_voxel_mesher_tetra_transition_toggle() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_warp_enabled(false);

	const VoxelMesher::Output stitched = build_plane(mesher, Vector3i(0, 0, 0), 0, 1.3f);
	mesher->set_lod_transitions_enabled(false);
	const VoxelMesher::Output unstitched = build_plane(mesher, Vector3i(0, 0, 0), 0, 1.3f);
	ZN_TEST_ASSERT(stitched.surfaces.size() == 1);
	ZN_TEST_ASSERT(unstitched.surfaces.size() == 1);
	const PackedVector3Array stitched_vertices = stitched.surfaces[0].arrays[Mesh::ARRAY_VERTEX];
	const PackedVector3Array unstitched_vertices = unstitched.surfaces[0].arrays[Mesh::ARRAY_VERTEX];
	const PackedFloat32Array lod_data = unstitched.surfaces[0].arrays[Mesh::ARRAY_CUSTOM0];
	ZN_TEST_ASSERT(unstitched_vertices.size() < stitched_vertices.size());
	for (int i = 3; i < lod_data.size(); i += 4) {
		uint32_t flags = 0;
		memcpy(&flags, &lod_data[i], sizeof(uint32_t));
		ZN_TEST_ASSERT(flags == 0);
	}
}

void test_voxel_mesher_tetra_geomorph_is_edge_bounded() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(2);
	mesher->set_warp_enabled(false);
	mesher->set_lod_transitions_enabled(false);
	mesher->set_lod_geomorph_enabled(true);
	const VoxelMesher::Output output = build_height_field(mesher, Vector3i(0, 0, 0), 0);
	ZN_TEST_ASSERT(output.surfaces.size() == 1);
	ZN_TEST_ASSERT(output.surfaces[0].blend_shapes.size() == 0);
	const PackedVector3Array vertices = output.surfaces[0].arrays[Mesh::ARRAY_VERTEX];
	const PackedFloat32Array timeline = output.surfaces[0].arrays[Mesh::ARRAY_CUSTOM2];
	ZN_TEST_ASSERT(timeline.size() == vertices.size() * 4);
	const float maximum_edge_length = Math::sqrt(3.f) * 0.5f;
	for (int i = 0; i < vertices.size(); ++i) {
		const Vector3 target(timeline[i * 4], timeline[i * 4 + 1], timeline[i * 4 + 2]);
		const uint32_t packed = static_cast<uint32_t>(timeline[i * 4 + 3]);
		ZN_TEST_ASSERT(packed != 0u);
		const float t_in = static_cast<float>(packed / 4096u) / 4095.f;
		const float t_out = static_cast<float>(packed % 4096u) / 4095.f;
		ZN_TEST_ASSERT(t_in >= 0.f);
		ZN_TEST_ASSERT(t_out <= 1.f + 1.0e-6f);
		ZN_TEST_ASSERT(t_in < t_out + 1.0e-3f);
		// Every slot's motion stays within its own lattice edge.
		ZN_TEST_ASSERT(vertices[i].distance_to(target) <= maximum_edge_length + 1.0e-5f);
	}
}

void test_voxel_mesher_tetra_geomorph_parent_identity() {
	Ref<VoxelMesherTetra> mesher;
	mesher.instantiate();
	mesher->set_subdivisions(1);
	mesher->set_warp_enabled(false);
	mesher->set_lod_transitions_enabled(false);

	const Vector3i parent_origin(0, 0, 0);
	const VoxelMesher::Output parent = build_parent_p1_field(mesher, parent_origin, 1);
	StdVector<Triangle> parent_triangles;
	append_triangles(parent, parent_origin, false, parent_triangles);
	ZN_TEST_ASSERT(!parent_triangles.empty());

	mesher->set_lod_geomorph_enabled(true);
	StdVector<Triangle> child_target_triangles;
	for (int z = 0; z < 2; ++z) {
		for (int y = 0; y < 2; ++y) {
			for (int x = 0; x < 2; ++x) {
				const Vector3i child_origin = parent_origin + Vector3i(x, y, z) * 4;
				const VoxelMesher::Output child = build_parent_p1_field(mesher, child_origin, 0);
				append_triangles(child, child_origin, true, child_target_triangles);
			}
		}
	}
	ZN_TEST_ASSERT(!child_target_triangles.empty());
	assert_points_on_surface(child_target_triangles, parent_triangles);
	assert_points_on_surface(parent_triangles, child_target_triangles);
}

} // namespace zylann::voxel::tests
