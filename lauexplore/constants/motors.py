"""
Defines allowed motor names for horizontal (X-axis) and vertical (Y-axis)
scanning directions.

These lists centralize the mapping between motor names used at the beamline
and the logical axes expected by the scan parser.

If additional motors become available, simply add them here without modifying
the scan logic elsewhere in the codebase.
"""

# Motors that correspond to horizontal motion (X direction)
X_MOTORS = [
    "xech",
    "hfoc", "dx",  
]

# Motors that correspond to vertical motion (Y direction)
Y_MOTORS = [
    "yech",
    "pfoc", "dy",   
]

# Optional: reverse lookup table (motor_name → axis_label)
# Useful for quickly determining whether a given motor belongs to X or Y.
AXIS_FROM_MOTOR = {
    **{motor: "x" for motor in X_MOTORS},
    **{motor: "y" for motor in Y_MOTORS},
}
