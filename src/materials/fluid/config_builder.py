import json


class SimConfig:
    def __init__(self, scene_file_path) -> None:
        self.config = None
        with open(scene_file_path, "r") as f:
            self.config = json.load(f)
        print(self.config)
    
    def get_cfg(self, name, default_value=None):
        if name not in self.config["Configuration"]:
            if default_value is not None:
                return default_value
            else:
                return None
        return self.config["Configuration"][name]
    
    def get_rigid_bodies(self):
        if "RigidBodies" in self.config:
            return self.config["RigidBodies"]
        else:
            return []
    
    def get_rigid_blocks(self):
        if "RigidBlocks" in self.config:
            return self.config["RigidBlocks"]
        else:
            return []
        
    def get_fluid_bodies(self):
        if "FluidBodies" in self.config:
            return self.config["FluidBodies"]
        else:
            return []
        
    def get_fluid_blocks(self):
        if "FluidBlocks" in self.config:
            return self.config["FluidBlocks"]
        else:
            return []
        
