#class for holding a single substance and its properties

class substance:

    def __init__(self, name, freqs, radiation_pattern) -> None:
        self.name = name
        self.radiation_pattern = radiation_pattern
        self.freqs = freqs

    def calculate_radiation(self, light_source):
        return self.radiation_pattern * light_source.radiation_pattern