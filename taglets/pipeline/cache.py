import logging

log = logging.getLogger(__name__)


class Cache:
    CACHE = {}

    @staticmethod
    def set(key, classes, data):
        log.info("Setting cache key %s", key)
        Cache.CACHE[key] = (classes, data)

    @staticmethod
    def get(key, classes):
        if key in Cache.CACHE:
            saved_classes, data = Cache.CACHE
            if saved_classes == classes:
                log.info("Cache hit")
                return data
            else:
                log.info("Cache miss (classes mismatch) for key %s", key)
                del Cache.CACHE[key]
                return None
        else:
            log.info("Cache miss (missing key %s)", key)
            return None
