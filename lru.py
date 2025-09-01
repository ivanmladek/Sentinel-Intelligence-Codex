from collections import OrderedDict


class LRUCache: 

    
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key, last=False)
        return self.cache[key]

    def put(self, key, value): 
        if key in self.cache:
            self.cache.move_to_end(key, last=False)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=True)

class AnotherLRUCache(LRUCache):
    
    def __init__(self, capacity):
        self.cache = OrderedDict()


cache =  LRUCache(2)
cache.put(1,1)
print(cache.cache)
cache.put(2,2)
print(cache.cache)
cache.get(1)


