import math


def front_distance(ranges, angle_min, angle_increment, range_min, range_max):
    """Return the minimum valid range in the forward 60-degree sector."""
    nearest = math.inf
    for index, distance in enumerate(ranges):
        angle = math.remainder(angle_min + index * angle_increment, math.tau)
        if abs(angle) <= math.pi / 6 and math.isfinite(distance):
            if range_min <= distance <= range_max:
                nearest = min(nearest, distance)
    return nearest
