# Automatically reload all of the modules in the current directory
# or its subdirectories every time you enter a command in IPython. 
# Usage: from the IPython toplevel, do
# In [1]: import autoreload
# autoreload enabled
# Author: Andrew Owens
import os, sys, traceback

def relative_module(m):
  return hasattr(m, '__file__') \
         and ((not m.__file__.startswith('/')) \
              or m.__file__.startswith(os.getcwd()))

def reload_all():
  # Reloading __main__ is supposed to throw an error
  # For some reason in ipython I did not get an error and lost the
  # ability to send reload_all() to my ipython shell after making
  # changes.
  excludes = set(['__main__', 'autoreload'])
  for name, m in list(sys.modules.iteritems()):
    # todo: Check source modification time (use os.path.getmtime) to see if it has changed.
    if m and relative_module(m) and (name not in excludes) and (not hasattr(m, '__no_autoreload__')):
      reload(m)
      #superreload.superreload(m)

def ipython_autoreload_hook(self):
  try:
    reload_all()
  except:
    traceback.print_exc()
    print 'Reload error. Modules not reloaded.'

def enable():
  try:
    import IPython
    ip = IPython.get_ipython()
    prerun_hook_name = 'pre_run_code_hook'
  except:
    try:
      # IPython 0.11
      import IPython.core.ipapi as ipapi
      prerun_hook_name = 'pre_run_code_hook'
      ip = ipapi.get()
    except:
      # IPython 0.10.1
      import IPython.ipapi as ipapi
      prerun_hook_name = 'pre_runcode_hook'    
      ip = ipapi.get()
  ip.set_hook(prerun_hook_name, ipython_autoreload_hook)
  print 'autoreload enabled'

# # taken from ipy_autoreload.py; should probably just use that instead
# def superreload(module, reload=reload):
#   """Enhanced version of the builtin reload function.
#   superreload replaces the class dictionary of every top-level
#   class in the module with the new one automatically,
#   as well as every function's code object.
#   """
#   module = reload(module)
#   # iterate over all objects and update them
#   count = 0
#   for name, new_obj in module.__dict__.items():
#     key = (module.__name__, name)
#     if _old_objects.has_key(key):
#       for old_obj in _old_objects[key]:
#         if type(new_obj) == types.ClassType:
#           old_obj.__dict__.update(new_obj.__dict__)
#           count += 1
#         elif type(new_obj) == types.FunctionType:
#           update_function(old_obj,
#                  new_obj,
#                  "func_code func_defaults func_doc".split())
#           count += 1
#         elif type(new_obj) == types.MethodType:
#           update_function(old_obj.im_func,
#                  new_obj.im_func,
#                  "func_code func_defaults func_doc".split())
#           count += 1
#     return module


# automatically enable when imported
__no_autoreload__ = True

#enable()


