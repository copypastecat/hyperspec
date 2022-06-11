#class for holding a single substance and its properties

class substance:

    def __init__(self, name, absorption_pattern) -> None:
        self.name = name
        self.absorption_pattern = absorption_pattern

    def calculate_radiation(self, light_source):
        pass