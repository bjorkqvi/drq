from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_time,
)


@add_datavar(name="eta")
@add_time()
class WavePlane(GriddedSkeleton):
    pass
