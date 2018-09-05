import pydash as _

def append_at(obj, path, val):
  if _.has(obj, path):
    _.update(obj, path, lambda coll: coll + [val])
  else:
    _.set_(obj, path, [val])
