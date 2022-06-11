#class for holding properties of a light source

class light_source:

    def __init__(self, name, radiation_patter) -> None:
        self.name = name
        self.radiation_pattern = radiation_patter