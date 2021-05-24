import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import Landmark
import pyvista as pv
import numpy as np
from typing import List, Tuple
import yaml
import os
import random

import OpenGL
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

random.seed(0)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


mesh_polygons = None
mesh_pts = None
gl_polygons = None

save_mesh_polygons = False

max_polygons = 1
more_polygons_per_frame = 20
polygon_count_change_direction = 1

if os.path.exists('/mesh'):
  if not os.path.exists('/mesh/polygons.yml'):
    save_mesh_polygons = True
  else:
    with open('/mesh/polygons.yml') as f:
      mesh_polygons = yaml.load(f, Loader=yaml.FullLoader)


# ids reference https://github.com/ManuelTS/augmentedFaceMeshIndices

mouth_pt_ids = [
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95
]

left_eye_pt_ids = [
  33, 246, 161, 160, 159, 158, 157, 173, 133,
  155, 154, 154, 145, 144, 163, 7
]

right_eye_pt_ids = [
  362, 398, 384, 385, 386, 387, 388, 466, 263,
  249, 390, 373, 374, 380, 381, 382
]

display = (1000, 1000)

def pv_polygons_to_polygons(pv_polygons):
  i = 0
  polygons = []
  while i < len(pv_polygons):
    n_pts = pv_polygons[i]
    polygons.append(tuple(pt for pt in pv_polygons[i+1:i+n_pts+1]))
    i += n_pts + 1
  return polygons

def draw_mesh(pts, polygons):
  global max_polygons, more_polygons_per_frame, polygon_count_change_direction
  gl.glTranslatef(-1.2,1,-1)
  gl.glScalef(1.75,-2,-2)

  # gl.glColor3f(0.0, 0.0, 1.0)
  # gl.glBegin(gl.GL_TRIANGLES)
  # for j in range(min(len(polygons), max_polygons)):
  #   for i in range(len(polygons[j])):
  #     vertex = pts[polygons[j][i]]
  #     gl.glVertex3f(vertex[0], vertex[1], vertex[2])
  # gl.glEnd()

  gl.glColor3f(0.0, 1.0, 0.0)
  gl.glBegin(gl.GL_LINES)
  for j in range(min(len(polygons), max_polygons)):
    for i in range(len(polygons[j]) - 1):
      start_point = pts[polygons[j][i]]
      end_point = pts[polygons[j][i+1]]
      gl.glVertex3f(start_point[0], start_point[1], start_point[2])
      gl.glVertex3f(end_point[0], end_point[1], end_point[2])
    start_point = pts[polygons[j][len(polygons[j]) - 1]]
    end_point = pts[polygons[j][0]]
    gl.glVertex3f(start_point[0], start_point[1], start_point[2])
    gl.glVertex3f(end_point[0], end_point[1], end_point[2])
  gl.glEnd()

  if polygon_count_change_direction > 0:
    max_polygons += more_polygons_per_frame
    max_polygons = min(max_polygons, len(polygons))
  else:
    max_polygons -= more_polygons_per_frame
    max_polygons = max(max_polygons, 0)


def update_mesh():
  global mesh_polygons, mesh_pts, cap, mouth_pt_ids, save_mesh_polygons
  success, image = cap.read()
  if not success:
    return
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference.
  image.flags.writeable = False
  results = face_mesh.process(image)

  # Draw the face mesh annotations on the image.
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
      points = [[landmark.x,landmark.y,landmark.z] for landmark in face_landmarks.landmark]
      # surf = cloud.delaunay_2d(alpha=1.0)
      # min_x = np.min([pt[0] for pt in points])
      # min_y = np.min([pt[1] for pt in points])
      # min_z = np.min([pt[2] for pt in points])
      # max_x = np.max([pt[0] for pt in points])
      # max_y = np.max([pt[1] for pt in points])
      # max_z = np.max([pt[2] for pt in points])
      # print(f'x({min_x},{max_x})')
      # print(f'y({min_y},{max_y})')
      # print(f'z({min_z},{max_z})')
      mesh_pts = points
      if mesh_polygons is None:
        cloud = pv.PolyData(points)
        exclude_polygons = pv.PolyData(points)
        exclude_polygons.faces = np.array(
          [len(mouth_pt_ids),] + mouth_pt_ids +
          [len(left_eye_pt_ids),] + left_eye_pt_ids +
          [len(right_eye_pt_ids),] + right_eye_pt_ids
          )
        surf = cloud.delaunay_2d(alpha=1.0, edge_source=exclude_polygons)
        mesh_polygons = surf.faces
        if save_mesh_polygons:
          with open('/mesh/polygons.yml', 'w') as f:
            yaml.dump(mesh_polygons.tolist(), f)
            save_mesh_polygons = False
      # print(dir(surf))
      # print(surf.faces[:10])
      # mp_drawing.draw_landmarks(
      #     image=image,
      #     landmark_list=face_landmarks,
      #     connections=mp_face_mesh.FACE_CONNECTIONS,
      #     landmark_drawing_spec=drawing_spec,
      #     connection_drawing_spec=drawing_spec)
      glut.glutPostRedisplay()
  

with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:


      glut.glutInit() # Initialize a glut instance which will allow us to customize our window
      glut.glutInitDisplayMode(glut.GLUT_RGBA) # Set the display mode to be colored
      glut.glutInitWindowSize(display[0], display[1])   # Set the width and height of your window
      glut.glutInitWindowPosition(0, 0)   # Set the position at which this windows should appear
      wind = glut.glutCreateWindow("OpenGL Coding Practice") # Give your window a title

      # Here is my keyboard input code
      def buttons(key,x,y):
        global wind, max_polygons, polygon_count_change_direction
        if key == b'q':
            glut.glutDestroyWindow(wind)
        if key == b'w':
          polygon_count_change_direction *= -1

      # ---Section 1---
      def lines(pts: List[Tuple]):
        # We have to declare the points in this sequence: bottom left, bottom right, top right, top left
        gl.glBegin(gl.GL_LINES) # Begin the sketch
        for i in range(len(pts) - 1):
          start_point = pts[i]
          end_point = pts[i+1]
          gl.glVertex3f(start_point[0], start_point[1], start_point[2])
          gl.glVertex3f(end_point[0], end_point[1], end_point[2])
        gl.glEnd() # Mark the end of drawing

      def init_perspective():
        gl.glViewport(0, 0, display[0], display[1])
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        # gl.glOrtho(0.0, 500, 0.0, 500, 0.0, 1.0)
        glu.gluPerspective(60, (display[0]/display[1]), 0.1, 500.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

      def showScreen():
        global mesh_polygons, mesh_pts, gl_polygons, max_polygons
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()
        init_perspective()
        if mesh_polygons is not None and mesh_pts is not None:
          if gl_polygons is None:
            gl_polygons = pv_polygons_to_polygons(mesh_polygons).copy()
            random.shuffle(gl_polygons)

          draw_mesh(mesh_pts, gl_polygons)
        glut.glutSwapBuffers()



      glut.glutDisplayFunc(showScreen)  # Tell OpenGL to call the showScreen method continuously
      glut.glutIdleFunc(update_mesh)     # Draw any graphics or shapes in the showScreen function at all times
      glut.glutKeyboardFunc(buttons)
      glut.glutMainLoop()

cap.release()