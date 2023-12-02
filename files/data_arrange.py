class ArrangeData:
    def __init__(self, x_value, y_value):
        self.x_value = x_value
        self.y_value = y_value

        self.arrange_points()

    @staticmethod
    def _sort_points_by_x(x_p, y_p):
        sorted_points = sorted(zip(x_p, y_p), key=lambda point: point[0])
        sorted_x, sorted_y = zip(*sorted_points)
        return sorted_x, sorted_y

    def arrange_points(self):
        x_points, y_points = self._sort_points_by_x(self.x_value, self.y_value)

        x = [x_points[0]]
        y = [y_points[0]]
        for i in range(1, len(y_points)):
            if y[-1] > y_points[i]:
                pass
            else:
                x.append(x_points[i])
                y.append(y_points[i])

        result_dict = {}
        for idx, value in enumerate(x):
            if value not in result_dict:
                result_dict[value] = y[idx]
            else:
                result_dict[value] = max(result_dict[value], y[idx])

        self.x_value = list(result_dict.keys())
        self.y_value = list(result_dict.values())

    def get_arranged_points(self):
        return self.x_value, self.y_value
