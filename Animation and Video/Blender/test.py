import bpy.ops

import bpy
from bpy import context

# Get the current scene
scene = context.scene

# Get the 3D cursor location
#cursor = scene.cursor.location

# Get the active object (assume we have one)
obj = context.active_object


initial_height = obj.location[2]
jump_height = 10
def jump_start(actFrame):
    obj.location[2] = initial_height
   # obj.keyframe_delete(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z
    obj.keyframe_insert(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z
    curr_kf_index = -1#3
    kf = obj.animation_data.action.fcurves[2].keyframe_points[curr_kf_index]
    print(kf.co)
    kf.interpolation = 'QUAD'
    kf.easing = 'EASE_OUT'

def jump_peak(actFrame):
    obj.location[2] = initial_height + jump_height
    #obj.keyframe_delete(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z
    obj.keyframe_insert(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z
    curr_kf_index = -1#4
    kf = obj.animation_data.action.fcurves[2].keyframe_points[curr_kf_index]
    print(kf.co)
    kf.interpolation = 'QUAD'
    kf.easing = 'EASE_IN'



def jump_end(actFrame):
    obj.location[2] = initial_height
    obj.keyframe_delete(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z
    obj.keyframe_insert(data_path='location', index=2, frame=actFrame) # index is of the position x, y, z

print(len( obj.animation_data.action.fcurves[2].keyframe_points ))
jump_start(90)
print(len( obj.animation_data.action.fcurves[2].keyframe_points ))
jump_peak(100)
jump_end(110)
print(len( obj.animation_data.action.fcurves[2].keyframe_points ))
