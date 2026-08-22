#include "test_voxel_mesher_tetra.h"
#include "../../meshers/tetra/voxel_mesher_tetra.h"
#include "../../storage/voxel_buffer.h"
#include "../../util/testing/test_macros.h"

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

} // namespace zylann::voxel::tests
