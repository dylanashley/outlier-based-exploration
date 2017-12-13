# -*- coding: ascii -*-

import abc
import itertools
import numpy as np

from scale import scale

__all__ = ['LOFCalculator', 'GridLOFCalculator']

EPS = 1e-9


class AbstractLOFCalculator(metaclass=abc.ABCMeta):
    def __len__(self):
        return len(self._points)

    def _is_valid_point(self, point):
        """Return validity of point if internal structure is correct."""
        if not len(self._points[0]) == len(point):
            return False
        for i in range(len(self._points[0])):
            if not 0 <= point[i] <= self.ranges[i]:
                return False
        return True

    def _is_valid_point_id(self, point_id):
        """Return validity of point if internal structure is correct."""
        return len(self._points) > point_id

    @abc.abstractmethod
    def _kNN(self, point_id):
        pass

    def _point_distance(self, first_id, second_id):
        assert (self._is_valid_point_id(first_id))
        assert (self._is_valid_point_id(second_id))

        # get euclidean distance between points
        rv = 0
        for i in range(len(self._points[0])):
            rv += (self._points[first_id][i] - self._points[second_id][i])**2
        return np.sqrt(rv)

    def _point_k_distance(self, point_id):
        assert (self._is_valid_point_id(point_id))

        # get distance to farthest kNN
        max_distance = 0
        for other_id in self._kNN(point_id):
            distance = self._point_distance(point_id, other_id)
            max_distance = max(max_distance, distance)
        return max_distance

    def _lof(self, point_id):
        assert (self._is_valid_point_id(point_id))

        # use equation to calculate local outlier factor
        rv = 0
        for other_id in self._kNN(point_id):
            rv += self._lrd(other_id)
        return rv / max(self.k * self._lrd(point_id), EPS)

    def _lrd(self, point_id):
        assert (self._is_valid_point_id(point_id))

        # use equation to calculate local reachability density
        kNN = self._kNN(point_id)
        rv = 0
        for other_id in kNN:
            distance = self._point_distance(point_id, other_id)
            k_distance = self._point_k_distance(other_id)
            rv += max(distance, k_distance)
        return self.k / max(rv, EPS)

    @abc.abstractmethod
    def insert(self, point, rv=None):
        pass


class LOFCalculator(AbstractLOFCalculator):
    def __init__(self, k, init_points, ranges=None):
        assert (k > 0)
        assert (len(init_points) > k)
        for i in range(1, len(init_points)):
            assert (len(init_points[0]) == len(init_points[i]))
            for j in range(len(init_points[0])):
                if ranges is not None:
                    assert (0 <= init_points[i][j] <= ranges[j])
                else:
                    assert (0 <= init_points[i][j] <= 1)

        # save args
        self.k = k
        if ranges is None:
            self.ranges = tuple((1 for _ in range(len(init_points[0]))))
        else:
            self.ranges = ranges

        # add initial points
        self._points = list()
        for point in init_points:
            self._points.append(point)

    def _kNN(self, point_id):
        assert (self._is_valid_point_id(point_id))

        # extract kNN from candidate points
        rv = sorted(
            [
                other_id for other_id in range(len(self._points))
                if other_id != point_id
            ],
            key=lambda other_id: self._point_distance(point_id, other_id))
        return set(rv[:self.k])

    def insert(self, point, rv=None):
        assert (self._is_valid_point(point))
        assert (rv in {'k-distance', 'LOF', None})

        point_id = len(self._points)
        self._points.append(point)

        if rv == 'k-distance':
            # return k-distance of new point
            return self._point_k_distance(point_id)
        elif rv == 'LOF':
            # return current local outlier factor of new point
            return self._lof(point_id)
        else:
            return


class GridLOFCalculator(AbstractLOFCalculator):
    def __init__(self, card, k, init_points, ranges=None):
        assert (k > 0)
        assert (len(init_points) > k)
        assert (all(c > 0 for c in card))
        for point in init_points:
            assert (len(card) == len(point))
            for i in range(len(card)):
                if ranges is not None:
                    assert (0 <= point[i] <= ranges[i])
                else:
                    assert (0 <= point[i] <= 1)

        # save args
        self.card = card
        self.k = k
        if ranges is None:
            self.ranges = tuple((1 for _ in range(len(self.card))))
        else:
            self.ranges = ranges

        # build cell table
        self._cells = np.empty(self.card, dtype=object)
        for cell in self._cell_iter():
            self._cells[cell] = {
                'points': set(),
                'poc': None,
                'k_distance': None,
                'kRNN_candidate_cells': set()
            }

        # add initial points
        self._points = list()
        for point in init_points:
            point_id = len(self._points)
            self._points.append(point)
            self._cells[self._point_to_cell(point_id)]['points'].add(point_id)

        # precompute all pairwise distances of cells
        for cell in self._cell_iter():
            self._cells[cell]['poc'] = self._proximally_ordered_cells(cell)

        # compute k-distances for cells
        for cell in self._cell_iter():
            self._cells[cell]['k_distance'] = self._cell_k_distance(cell)

        # compute kRNN for cells
        for cell in self._cell_iter():
            for i in range(self._cells[cell]['k_distance'] + 1):
                for other_cell in self._cells[cell]['poc'][i]:
                    self._cells[other_cell]['kRNN_candidate_cells'].add(cell)

    def _all_cells_in_hypercube(self, top_left, bottom_right):
        assert (self._is_valid_cell(top_left))
        assert (self._is_valid_cell(bottom_right))
        for i in range(len(self.card)):
            assert (top_left[i] <= bottom_right[i])

        # compute the distance between the cells on each axis
        difference = (bottom_right[i] - top_left[i]
                      for i in range(len(self.card)))

        # compute all cells between
        rv = set()
        for translation in itertools.product(*(range(v + 1)
                                               for v in difference)):
            cell = tuple((top_left[i] + d for i, d in enumerate(translation)))
            rv.add(cell)
        return rv

    def _cell_iter(self):
        """Helper function to iterate linearly through matrix."""
        return itertools.product(*(range(v) for v in self.card))

    def _cell_k_distance(self, start_cell):
        """Calculate an upper bound for the number of additional cells
        in each direction that must be searched to find the kNN
        for any point that will ever be in the starting cell.
        """
        assert (self._is_valid_cell(start_cell))

        point_count = max(len(self._cells[start_cell]['points']) - 1, 0)
        i = 1
        while point_count < self.k:
            for cell in self._cells[start_cell]['poc'][i]:
                point_count += len(self._cells[cell]['points'])
            i += 1  # end with increment to handle corner points
        return min(i + 1, len(self._cells[start_cell]['poc']) - 1)

    def _is_valid_cell(self, cell):
        """Return validity of cell if internal structure is correct."""
        if len(cell) != len(self.card):
            return False
        for i in range(len(self.card)):
            if (0 > cell[i]) or (cell[i] >= self.card[i]):
                return False
        return True

    def _kNN(self, point_id):
        assert (self._is_valid_point_id(point_id))

        # get all points in cells that were precomputed as candidates
        cell = self._point_to_cell(point_id)
        rv = self._cells[cell]['points'] - {point_id}
        for i in range(1, self._cells[cell]['k_distance'] + 1):
            for other_cell in self._cells[cell]['poc'][i]:
                rv.update(self._cells[other_cell]['points'])

        # extract kNN from candidate points
        rv = sorted(
            list(rv),
            key=lambda other_id: self._point_distance(point_id, other_id))
        return set(rv[:self.k])

    def _point_to_cell(self, point_id):
        assert (self._is_valid_point_id(point_id))

        rv = list()
        for i in range(len(self.card)):
            coord = int(
                scale(self._points[point_id][i], 0, self.ranges[i], 0,
                      self.card[i]))
            coord = min(coord, self.card[i] - 1)
            rv.append(coord)
        return tuple(rv)

    def _proximally_ordered_cells(self, start_cell):
        assert (self._is_valid_cell(start_cell))

        # create points to define current hypercube to search
        top_left = list(start_cell)
        bottom_right = list(start_cell)

        # search until found distance to all cells
        rv = list()
        found = set()
        while len(found) != np.prod(self.card):
            cells = self._all_cells_in_hypercube(
                tuple(top_left), tuple(bottom_right))
            rv.append(cells - found)

            # expand hypercube by one in each direction
            for i in range(len(start_cell)):
                top_left[i] = max(top_left[i] - 1, 0)
                bottom_right[i] = min(bottom_right[i] + 1, self.card[i] - 1)

            # update cells found so far
            found.update(rv[-1])

        # cardinality should never change so return tuple
        return tuple(rv)

    def insert(self, point, rv=None):
        assert (self._is_valid_point(point))
        assert (rv in {'k-distance', 'LOF', None})

        point_id = len(self._points)
        self._points.append(point)
        cell = self._point_to_cell(point_id)
        self._cells[cell]['points'].add(point_id)

        # fix data structures
        for other_cell in self._cells[cell]['kRNN_candidate_cells']:
            old_k_distance = self._cells[other_cell]['k_distance']
            new_k_distance = self._cell_k_distance(other_cell)
            for k_distance in range(new_k_distance + 1, old_k_distance):
                for distant_cell in self._cells[other_cell]['poc'][k_distance]:
                    self._cells[distant_cell]['kRNN_candidate_cells'].remove(
                        other_cell)
            self._cells[other_cell]['k_distance'] = new_k_distance

        if rv == 'k-distance':
            # return k-distance of new point
            return self._point_k_distance(point_id)
        elif rv == 'LOF':
            # return current local outlier factor of new point
            return self._lof(point_id)
        else:
            return
