from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.append("./")

from src.pepNmemb.scripts.get_insertion import (
    get_closest_lipid_z,
    get_index_shortest_distance,
)


class TestGetIndexShortestDistance(unittest.TestCase):

    def test_basic_functionality(self):
        point_p = np.array([[1, 2, 3]])
        b = np.array(
            [
                [4, 5, 6],  # distance = 5.196
                [2, 3, 4],  # distance = 1.732
                [10, 10, 10],  # distance = 13.856
            ]
        )

        expected_closest_point = np.array([2, 3, 4])
        expected_distance = np.sqrt(3)

        result_point, result_distance = get_index_shortest_distance(point_p, b)

        np.testing.assert_array_almost_equal(result_point, expected_closest_point)
        self.assertAlmostEqual(result_distance, expected_distance)

    def test_with_identical_point(self):
        point_p = np.array([[1, 2, 3]])
        b = np.array(
            [[4, 5, 6], [1, 2, 3], [10, 10, 10]]  # This is identical to point_p
        )

        expected_closest_point = np.array([4, 5, 6])
        expected_distance = np.sqrt(27)

        result_point, result_distance = get_index_shortest_distance(point_p, b)

        np.testing.assert_array_almost_equal(result_point, expected_closest_point)
        self.assertAlmostEqual(result_distance, expected_distance)

    def test_with_empty_array(self):
        # Test with empty array - this should raise an error
        point_p = np.array([[1, 2, 3]])
        b = np.array([])

        with self.assertRaises(Exception):
            get_index_shortest_distance(point_p, b)

    def test_with_single_point(self):
        point_p = np.array([[1, 2, 3]])
        b = np.array([[4, 5, 6]])

        expected_closest_point = np.array([4, 5, 6])
        expected_distance = np.sqrt(27)

        result_point, result_distance = get_index_shortest_distance(point_p, b)

        np.testing.assert_array_almost_equal(result_point, expected_closest_point)
        self.assertAlmostEqual(result_distance, expected_distance)

    def test_with_all_identical_points(self):
        point_p = np.array([[1, 2, 3]])
        b = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

        with self.assertRaises(Exception):
            get_index_shortest_distance(point_p, b)


class TestGetClosestLipidZ(unittest.TestCase):

    def setUp(self):
        # Common test data
        self.p_up = 20.0
        self.p_low = -20.0

        # Sample residue position
        self.res_pos = np.array([[5.0, 5.0, 0.0]])

        # Sample lipids array with 3D coordinates
        self.lipids = np.array(
            [
                [1.0, 1.0, 15.0],  # Upper layer lipid
                [10.0, 10.0, -15.0],  # Lower layer lipid
                [6.0, 6.0, 2.0],  # Near the residue
                [100.0, 100.0, 25.0],  # Far away lipid
            ]
        )

    def test_finds_closest_lipid_first_radius(self):
        result = get_closest_lipid_z(self.res_pos, self.lipids, self.p_up, self.p_low)

        self.assertEqual(result, 2.0)

    def test_finds_lipid_in_second_radius(self):
        res_pos = self.res_pos
        lipids = np.array(
            [
                [15.0, 15.0, 15.0],  # Outside first radius but inside second
                [100.0, 100.0, 100.0],  # Far away lipid
            ]
        )

        # Mock KDTree to simulate empty result for first radius but hit for second
        with patch("scipy.spatial.KDTree") as mock_kdtree:
            mock_instance = MagicMock()
            mock_kdtree.return_value = mock_instance

            # First radius (20) returns empty, second radius (40) returns hit
            mock_instance.query_ball_point.side_effect = [
                [[]],  # Empty result for radius 20
                [[0]],  # First lipid found for radius 40
                [[0]],  # Not needed but included for completeness
                [[0, 1]],  # Not needed but included for completeness
                [[0, 1]],  # Not needed but included for completeness
                [[0, 1]],  # Not needed but included for completeness
            ]

            # Mock the get_index_shortest_distance function
            with patch(
                "pepNmemb.scripts.get_insertion.get_index_shortest_distance"
            ) as mock_get_index:
                mock_get_index.return_value = (lipids[0], 0.0)

                result = get_closest_lipid_z(res_pos, lipids, self.p_up, self.p_low)

                # Should return z-coordinate of the first lipid (15.0)
                self.assertEqual(result, 15.0)

    def test_no_lipids_found(self):
        far_lipids = np.array([[1000.0, 1000.0, 1000.0], [2000.0, 2000.0, 2000.0]])

        # Mock KDTree to return empty results for all radii
        with patch("scipy.spatial.KDTree") as mock_kdtree:
            mock_instance = MagicMock()
            mock_kdtree.return_value = mock_instance

            # All calls to query_ball_point return empty results
            mock_instance.query_ball_point.return_value = [[]]

            result = get_closest_lipid_z(
                self.res_pos, far_lipids, self.p_up, self.p_low
            )

            # Should return None as no lipids were found
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
