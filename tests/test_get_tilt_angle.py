from __future__ import annotations

import math as m
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
sys.path.append("./")

from src.pepNmemb.scripts.get_tilt_angle import (
    angle_between_vectors,
    dot,
    find_consecutive_sublists,
    get_coords_spanning_memb,
    get_vector,
    relative_tilt,
    vector_length,
)





class TestConsecutiveSublists(unittest.TestCase):
    """
    Test cases for the find_consecutive_sublists function
    """

    def test_empty_list(self):

        self.assertEqual(find_consecutive_sublists([]), [])

    def test_no_consecutive_groups(self):

        self.assertEqual(find_consecutive_sublists([1, 3, 5, 7, 9]), [])
        self.assertEqual(find_consecutive_sublists([1, 2, 3, 5, 7]), [])

    def test_one_consecutive_group(self):

        self.assertEqual(find_consecutive_sublists([1, 2, 3, 4, 5]), [[1, 2, 3, 4, 5]])

    def test_multiple_consecutive_groups(self):

        self.assertEqual(
            find_consecutive_sublists([1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 15]),
            [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]],
        )

    def test_unsorted_input(self):

        self.assertEqual(
            find_consecutive_sublists([5, 1, 2, 3, 4, 10, 7, 11, 8, 9]),
            [[1, 2, 3, 4, 5], [7, 8, 9, 10, 11]],
        )


class TestVectorOperations(unittest.TestCase):

    def test_get_vector(self):
        coord1 = (5, 3, 7)
        coord2 = (2, 1, 4)
        expected = (3, 2, 3)
        self.assertEqual(get_vector(coord1, coord2), expected)

    def test_vector_length(self):
        v = (3, 4, 5)
        expected = m.sqrt(3**2 + 4**2 + 5**2)
        self.assertAlmostEqual(vector_length(v), expected)

    def test_dot_product(self):
        v1 = (1, 2, 3)
        v2 = (4, 5, 6)
        expected = 1 * 4 + 2 * 5 + 3 * 6
        self.assertEqual(dot(v1, v2), expected)

    def test_angle_between_vectors(self):
        # Parallel vectors
        v1 = (1, 0, 0)
        v2 = (2, 0, 0)
        self.assertAlmostEqual(angle_between_vectors(v1, v2), 0)

        # Perpendicular vectors
        v1 = (1, 0, 0)
        v2 = (0, 1, 0)
        self.assertAlmostEqual(angle_between_vectors(v1, v2), 90)

        # 45 degree angle
        v1 = (1, 0, 0)
        v2 = (1, 1, 0)
        self.assertAlmostEqual(angle_between_vectors(v1, v2), 45)


class TestGetCoordsSpanningMemb(unittest.TestCase):

    @patch("MDAnalysis.Universe")
    def test_residue_filtering(self, mock_universe):
        """Test filtering of residues based on z-coordinate"""
        # Mock the selection of atoms
        mock_universe.select_atoms = MagicMock()

        # Setup mock positions for different residues
        residue_positions = {
            1: np.array([[0, 0, 5]]),  # Within range (p_up + 10 > 5 > p_up - 6)
            2: np.array([[0, 0, 6]]),  # Within range
            3: np.array([[0, 0, 7]]),  # Within range
            4: np.array([[0, 0, 8]]),  # Within range
            5: np.array([[0, 0, 9]]),  # Within range
            6: np.array([[0, 0, 20]]),  # Outside range (too high)
            7: np.array([[0, 0, -10]]),  # Outside range (too low)
        }

        def mock_select_atoms(query):
            # Extract residue ID from the query
            res_id = int(query.split()[-1])
            result = MagicMock()
            result.positions = residue_positions.get(res_id, np.array([[1,1,2]]))
            return result

        mock_universe.select_atoms.side_effect = mock_select_atoms

        # Test with p_up = 0 (should select residues 1-5 with z between -6 and 10)
        residues = (0, 7)  # Residue IDs 0-7
        p_up = 0

        _, selected_res = get_coords_spanning_memb(mock_universe, residues, p_up)

        # We expect residues 1, 2, 3, 4, 5 to be selected (after adjusting for the +1 in the function)
        expected_selected = [1, 2, 3, 4, 5, 6]  # +1 is added in the function
        self.assertEqual(selected_res, expected_selected)


class TestRelativeTilt(unittest.TestCase):
    """Test cases for relative_tilt function"""

    @patch("MDAnalysis.analysis.leaflet.LeafletFinder")
    def test_relative_tilt(self, mock_leaflet_finder):
        # Create a mock Peptide object
        mock_peptide = MagicMock()
        mock_peptide.u = MagicMock()
        mock_peptide.pep_name = "test_peptide"
        mock_peptide.pep_dict = {"pep1": (1, 10)}

        # Mock the trajectory
        mock_trajectory = MagicMock()
        mock_trajectory.time = 1000  # 1 ns
        mock_peptide.u.trajectory = [mock_trajectory]
        mock_peptide.load_traj.return_value = 0, 1

        # Mock P atom positions
        p_positions = np.array([[0, 0, 5], [0, 0, -5]])
        mock_selection = MagicMock()
        mock_selection.positions = p_positions
        mock_peptide.u.select_atoms.return_value = mock_selection

        # Mock get_coords_spanning_memb
        with patch(
            "src.pepNmemb.scripts.get_tilt_angle.get_coords_spanning_memb"
        ) as mock_get_coords:
            # Return some coordinates for a peptide that spans the membrane
            mock_get_coords.return_value = (
                [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]],
                [2, 3, 4],
            )

            # Mock Line.best_fit from skspatial
            with patch("skspatial.objects.Line.best_fit") as mock_line_fit:
                mock_line = MagicMock()
                mock_line.vector = np.array([0, 0, 1])  # Aligned with membrane normal
                mock_line.point = np.array([0, 0, 0])
                mock_line_fit.return_value = mock_line

                # Mock DataFrame.to_csv
                with patch("pandas.DataFrame.to_csv") as mock_to_csv:
                    result = relative_tilt(mock_peptide, 2, "output_folder")  # 2 membrane lipids

                    # Check if the result is a DataFrame with expected columns
                    self.assertIsInstance(result, pd.DataFrame)
                    self.assertTrue(
                        all(
                            col in result.columns
                            for col in [
                                "Time (ns)",
                                "Peptide_num",
                                "Angle",
                                "len coords",
                            ]
                        )
                    )

                    # Check if to_csv was called
                    mock_to_csv.assert_called_once()


if __name__ == "__main__":
    unittest.main()
