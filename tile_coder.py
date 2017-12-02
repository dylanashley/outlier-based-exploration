# -*- coding: ascii -*-


class TileCoder:
    def __init__(self, card, n):
        self.card = card
        self.n = n
        self.tiling_area = self.card**2
        self.tile_count = self.tiling_area * self.n

    def discretize(self, x, y, indices=None):
        if indices is None:
            indices = np.zeros(self.n)
        else:
            assert (len(indices) == self.n)
        for tiling in range(self.n):

            offset = 0 if self.n == 1 else tiling / float(self.n)

            x_index = int(x * (self.card - 1) + offset)
            x_index = min(x_index, self.card - 1)

            y_index = int(y * (self.card - 1) + offset)
            y_index = min(y_index, self.card - 1)

            indices[
                tiling] = x_index + y_index * self.card + tiling * self.tiling_area
