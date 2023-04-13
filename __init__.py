

# Custom decorator handling read only
def _constant(f):
    def fset(self, value):
        raise TypeError("Constant is read only")

    def fget(self):
        return f()
    return property(fget, fset)


# All methods with an "@_constant" decorator above are read-only.
# Remove decorators to make constant assignable.
class _Const(object):
    @_constant
    def STANDARD_WEBCAM():
        return 0

    @_constant
    def CAPTURE_WEBCAM():
        return 0

    @_constant
    def CAPTURE_IMAGE():
        return 1

    @_constant
    def NORMAL_MODE():
        return 0

    @_constant
    def SHOWCASE_MODE():
        return 1

    @_constant
    def DEBUG_MODE():
        return 2

    @_constant
    def ALIGN_LEFT():
        return 0

    @_constant
    def ALIGN_RIGHT():
        return 1

    @_constant
    def ALIGN_TOP():
        return 2

    @_constant
    def ALIGN_BOTTOM():
        return 3

    @_constant
    def ALIGN_CENTER():
        return 4


# Assign class to variables
CONST = _Const()




