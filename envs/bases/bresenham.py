import numpy as np
import math
# import numba # not correct

class Bresenham:
    def __init__(self, geo_grid_data_):
        self.geo_grid_data = geo_grid_data_

    def can_see_eachother(self, x0, y0, x1, y1):
        init_x = x0
        init_y = y0
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        prev_x0 = x0
        prev_y0 = y0

        _return_value = True
        x_increased = 0
        y_increased = 0
        while True:
            _rgb_value = self.geo_grid_data.get_grid_RGB_property(x0, y0)
            if _rgb_value[1] > 10 or _rgb_value[0] > 10:  # G has value
                _return_value = False
                break

            x_increased = 0
            y_increased = 0

            # yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                x_increased = 1
                err -= dy
                prev_x0 = x0 #충돌직전의 위치로.
                x0 += sx
            if e2 < dx:
                y_increased = 1
                err += dx
                prev_y0 = y0 #충돌직전의 위치로.
                y0 += sy

        if x_increased == 1:
            x0 = prev_x0
        if y_increased == 1:
            y0 = prev_y0

        _rgb_value = self.geo_grid_data.get_grid_RGB_property(x0, y0)
        if _rgb_value[1] > 10 or _rgb_value[0] > 10:  # G has value
            print("장애물에 파묻힘..", x0, y0)

        return _return_value, (x0, y0) #20231101 return 값 추가. 장애물 충돌 직전의 위치를 return

    def do_raytracing_one_degree(self, x0_, y0_, range_, obs_visible_, degree):
        init_x = x0_
        init_y = y0_

        x1 = int(range_ * math.cos(math.radians(degree)) + x0_)
        y1 = int(range_ * math.sin(math.radians(degree)) + y0_)
        x1 = max(0, min(x1, self.geo_grid_data.width() - 1))
        y1 = max(0, min(y1, self.geo_grid_data.height() - 1))
        dx = abs(x1 - x0_)
        dy = abs(y1 - y0_)
        sx = 1 if x0_ < x1 else -1
        sy = 1 if y0_ < y1 else -1
        err = dx - dy

        # prev_x0 = x0_
        # prev_y0 = y0_

        _return_value = True
        while True:
            # print("do_raytracing i:", i, "x0:", x0_, " y0:", y0_, " x1:", x1, " y1:", y1)
            _rgb_value = self.geo_grid_data.get_grid_RGB_property(x0_, y0_)
            if _rgb_value[1] > 3 or _rgb_value[0] > 10:  # G has value
                _return_value = False
                offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                obs_visible_[offset_x + range_, offset_y + range_] = 2
                break
            # yield (x0, y0)
            if x0_ == x1 and y0_ == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                # prev_x0 = x0_ #충돌직전의 위치로.
                x0_ += sx
                # 장애물 시작점까지 보이도록
                offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                obs_visible_[offset_x + range_, offset_y + range_] = 1
            if e2 < dx:
                err += dx
                # prev_y0 = y0_ #충돌직전의 위치로.
                y0_ += sy
                # 장애물 시작점까지 보이도록
                offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                obs_visible_[offset_x+range_, offset_y+range_] = 1

    # @numba.jit(nopython=False, parallel=True) #not correct
    def do_raytracing(self, x0_, y0_, range_, obs_visible_, step):
        init_x = x0_
        init_y = y0_

        for i in range(0, 360, step): # degree
            x0_ = init_x
            y0_ = init_y
            x1 = int(range_ * math.cos(math.radians(i)) + x0_)
            y1 = int(range_ * math.sin(math.radians(i)) + y0_)
            x1 = max(0, min(x1, self.geo_grid_data.width() - 1))
            y1 = max(0, min(y1, self.geo_grid_data.height() - 1))
            dx = abs(x1 - x0_)
            dy = abs(y1 - y0_)
            sx = 1 if x0_ < x1 else -1
            sy = 1 if y0_ < y1 else -1
            err = dx - dy

            # prev_x0 = x0_
            # prev_y0 = y0_

            _return_value = True
            while True:
                # print("do_raytracing i:", i, "x0:", x0_, " y0:", y0_, " x1:", x1, " y1:", y1)
                _rgb_value = self.geo_grid_data.get_grid_RGB_property(x0_, y0_)
                if _rgb_value[1] > 3 or _rgb_value[0] > 10:  # G has value
                    _return_value = False
                    offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                    offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                    obs_visible_[offset_x + range_, offset_y + range_] = 1
                    break
                # yield (x0, y0)
                if x0_ == x1 and y0_ == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    # prev_x0 = x0_ #충돌직전의 위치로.
                    x0_ += sx
                    # 장애물 시작점까지 보이도록
                    offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                    offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                    obs_visible_[offset_x + range_, offset_y + range_] = 1
                if e2 < dx:
                    err += dx
                    # prev_y0 = y0_ #충돌직전의 위치로.
                    y0_ += sy
                    # 장애물 시작점까지 보이도록
                    offset_x = max(-range_, min(x0_ - init_x, range_ - 1))
                    offset_y = max(-range_, min(y0_ - init_y, range_ - 1))
                    obs_visible_[offset_x+range_, offset_y+range_] = 1

