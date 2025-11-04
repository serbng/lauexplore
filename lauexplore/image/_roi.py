from dataclasses import dataclass

@dataclass
class ROI:
    center: tuple[int, int]
    boxsize: tuple[float, float]
    
    @property
    def x0(self) -> int:
        return self.center[0]
    
    @property
    def y0(self) -> int:
        return self.center[1]
    
    @property
    def xboxsize(self) -> int:
        return self.boxsize[0]
    
    @property
    def yboxsize(self) -> int:
        return self.boxsize[1]
    
    @property
    def x1(self) -> int:
        return int(self.x0 - self.xboxsize//2)
    
    @property
    def x2(self) -> int:
        return int(self.x0 + self.xboxsize//2)
    
    @property
    def y1(self) -> int:
        return int(self.y0 - self.yboxsize//2)
    
    @property
    def y2(self) -> int:
        return int(self.y0 + self.yboxsize//2)
    
    @property
    def extent(self) -> tuple[int, int, int, int]:
        return (self.x1, self.x2, self.y1, self.y2)
    
    @property
    def corners(self) -> tuple[tuple[int, int]]:
        return ((self.x1, self.y1),
                (self.x1, self.y2),
                (self.x2, self.y2),
                (self.x2, self.y1))
    
    @property
    def path(self) -> tuple[tuple[int, int]]:
        return ((self.x1, self.y1),
                (self.x1, self.y2),
                (self.x2, self.y2),
                (self.x2, self.y1),
                (self.x1, self.y1))