#include "test_voxel_mesher_tetra.h"
#include "../../meshers/tetra/voxel_mesher_tetra.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/testing/test_macros.h"

#include <cstring>

namespace zylann::voxel::tests {

namespace {

VoxelMesher::Output build_plane(
		const Ref<VoxelMesherTetra> &mesher, Vector3i content_origin, uint8_t lod_index, float plane_height,
		bool collision_hint = false) {
	const int lod_stride = 1 << lod_index;
	const Vector3i padded_origin = content_origin - Vector3iUtil::create(2 * lod_stride);

	VoxelBuffer voxels(VoxelBuffer::ALLOCATOR_DEFAULT);
	voxels.set_channel_depth(VoxelBuffer::CHANNEL_SDF, VoxelBuffer::DEPTH_32_BIT);
	voxels.create(Vector3iUtil::create(8));
	for (int z = 0; z < 8; ++z) {
		for (int y = 0; y < 8; ++y) {
			for (int x = 0; x < 8; ++x) {
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
	voxels.create(Vector3iUtil::create(8));
	for (int z = 0; z < 8; ++z) {
		for (int y = 0; y < 8; ++y) {
			for (int x = 0; x < 8; ++x) {
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

} // namespace zylann::voxel::tests
