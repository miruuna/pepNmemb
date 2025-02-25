import unittest
import numpy as np
from pepNmemb.scripts.get_insertion import get_index_shortest_distance
from pepNmemb.core.classes import Peptide

class TestGetIndexShortestDistance(unittest.TestCase):
    
    def test_basic_functionality(self):
        point_p = np.array([[1, 2, 3]])  
        b = np.array([
            [4, 5, 6],    # distance = 5.196
            [2, 3, 4],    # distance = 1.732
            [10, 10, 10]  # distance = 13.856
        ])
        
        expected_closest_point = np.array([2, 3, 4])
        expected_distance = np.sqrt(3)  
        
        result_point, result_distance = get_index_shortest_distance(point_p, b)
        
        np.testing.assert_array_almost_equal(result_point, expected_closest_point)
        self.assertAlmostEqual(result_distance, expected_distance)
    
    def test_with_identical_point(self):
        point_p = np.array([[1, 2, 3]])
        b = np.array([
            [4, 5, 6],
            [1, 2, 3],  # This is identical to point_p
            [10, 10, 10]
        ])
        
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
        b = np.array([
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        ])
        
        with self.assertRaises(Exception):
            get_index_shortest_distance(point_p, b)

if __name__ == '__main__':
    unittest.main()