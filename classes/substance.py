#class for holding a single substance and its properties

class substance:

    def __init__(self, name, radiation_pattern) -> None:
        self.name = name
        self.radiation_pattern = radiation_pattern

    def calculate_radiation(self, light_source):
        return self.radiation_pattern * light_source.radiation_pattern