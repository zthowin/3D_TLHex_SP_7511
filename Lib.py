#----------------------------------------------------------------------------------------
# Module to add methods belonging to classElement from various module files.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    January 17, 2022
#----------------------------------------------------------------------------------------
def add_methods_from(*modules):
  def decorator(Class):
    for module in modules:
      for method in getattr(module, "__methods__"):
        setattr(Class, method.__name__, method)
    return Class
  return decorator

def register_method(methods):
  def register_method(method):
    methods.append(method)
    return method
  return register_method
