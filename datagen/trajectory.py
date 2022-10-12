from typing import Dict, Any


class Trajectory(object):
    def __init__(self, data):
        self.data: Dict[str, Any] = data

        assert 'x' in self.data.keys() or 'action' in self.data.keys(), 'rl traj should have "action", sl traj should have "x"'
        if 'action' in self.data.keys():
            self.length = len(self.data['action'])
        elif 'x' in self.data.keys():
            self.length = len(self.data['x'])
        else:
            raise ValueError("can't find 'action' or 'x' keys in trajectory data")

    def __len__(self):
        return self.length

    def __copy__(self):
        return Trajectory(dict(self.data))

    #    def __getitem__(self, item):
    #        return self.data.__getitem__(item)

    #    def __setitem__(self, key, value):
    #        self.data.__setitem__(key, value)