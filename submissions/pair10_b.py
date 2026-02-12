class Pt:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def dist_sq(self):
        return self.a * self.a + self.b * self.b
