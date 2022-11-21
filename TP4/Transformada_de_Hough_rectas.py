"""
IA TP4 Reconocimiento de rectas con transformada de Hough
"""

import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

###############################################################################################
###        Busca en la imagen con la formula para rectas de la transformada de Hough        ###
###############################################################################################
def busca_rectas(image, edge_image, num_rhos, num_thetas, bin_threshold):
  #tamaño de la imagen
  img_height, img_width = edge_image.shape[:2]
  img_height_half = img_height / 2
  img_width_half = img_width / 2
  
  # x cos θ + y sin θ = ρ
  # rangos de Rho y Theta
  diag_len = np.sqrt(np.square(img_height) + np.square(img_width))
  dtheta = 180 / num_thetas
  drho = (2 * diag_len) / num_rhos
  
  ## Thetas son contenedores creados de 0 a 180 grados con incremento proporcionado de dtheta.
  thetas = np.arange(0, 180, step=dtheta)
  
  ## Rho varía de -diag_len a diag_len donde diag_len es la longitud diagonal de la imagen de entrada.
  rhos = np.arange(-diag_len, diag_len, step=drho)
  
  # Calcula Cos (theta) y Sin (theta) se requerirá más adelante al calcular rho
  cos_thetas = np.cos(np.deg2rad(thetas))
  sin_thetas = np.sin(np.deg2rad(thetas))
  
  # Matriz de los acumuladores de Hough de theta vs rho.
  accumulator = np.zeros((len(rhos), len(thetas)))
  
  # Diagrama de espacio de Hough para la imagen.
  figure = plt.figure()
  hough_plot = figure.add_subplot()
  hough_plot.set_facecolor((0, 0, 0))
  hough_plot.title.set_text("Espacio de Hough")
  
  # Iterar a través de píxeles y si el píxel no es cero procesarlo por mucho espacio.
  for y in range(img_height):
    for x in range(img_width):
      if edge_image[y][x] != 0: #pixel blanco.
        edge_pt = [y - img_height_half, x - img_width_half]
        hough_rhos, hough_thetas = [], [] 
        
        # Iterar a través de rangos theta para calcular los valores rho
        for theta_idx in range(len(thetas)):
          # Calcular valores rho.
          rho = (edge_pt[1] * cos_thetas[theta_idx]) + (edge_pt[0] * sin_thetas[theta_idx])
          theta = thetas[theta_idx]
          
          # Obtener el índice del valor rho más cercano.
          rho_idx = np.argmin(np.abs(rhos - rho))
          
          #incrementar el voto por el par (rho_idx,theta_idx)
          accumulator[rho_idx][theta_idx] += 1
          
          # Agregamos valores de rho y theta en hough_rhos y hough_thetas respectivamente para el trazado del espacio de Hough.
          hough_rhos.append(rho)
          hough_thetas.append(theta)
        
        # Trazamos el espacio de Hough a partir de los valores.
        hough_plot.plot(hough_thetas, hough_rhos, color="white", alpha=0.05)
  
  # Imagen de salida con líneas detectadas dibujadas
  output_img = image.copy()
  # Lista de salida de líneas detectadas. Una sola línea sería una tupla de (rho,theta,x1,y1,x2,y2)
  out_lines = []
  
  for y in range(accumulator.shape[0]):
    for x in range(accumulator.shape[1]):
      # Si el número de votos es mayor que bin_threshold proporcionado, lo preselecciona como candidato.
      if accumulator[y][x] > bin_threshold:
        rho = rhos[y]
        theta = thetas[x]
        
        # a y b son intersecciones en las direcciones x e y
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        
        x0 = (a * rho) + img_width_half
        y0 = (b * rho) + img_height_half
        
        # Obtiene los puntos extremos para dibujar la línea.
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        # Plot the Maxima point on the Hough Space Plot
        hough_plot.plot([theta], [rho], marker='o', color="yellow")
        
        # Draw line on the output image
        output_img = cv2.line(output_img, (x1,y1), (x2,y2), (0,255,0), 1)
        
        # Add the data for the line to output list
        out_lines.append((rho,theta,x1,y1,x2,y2))

  # Show the Hough plot
  hough_plot.invert_yaxis()
  hough_plot.invert_xaxis()
  plt.show()
  
  return output_img, out_lines

###############################################################################################


def peak_votes(accumulator, thetas, rhos):
    """ Encuentra el número máximo de votos en el acumulador Hough """
    idx = np.argmax(accumulator)
    rho = rhos[int(idx / accumulator.shape[1])]
    theta = thetas[idx % accumulator.shape[1]]

    return idx, theta, rho


def theta2gradient(theta):
    """ Encuentra la pendiente m de theta """
    return np.cos(theta) / np.sin(theta)


def rho2intercept(theta, rho):
    """ Encuentra la intercepcion b con rho """
    return rho / np.sin(theta)

def main():
    
    img_path = 'rectas.png'
    num_rho = 180
    num_theta = 180
    bin_threshold = 150
    lines_are_white = True
    
     
    input_img = cv2.imread(img_path)
    
    #Detección de bordes en la imagen de entrada.
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    ret, edge_image = cv2.threshold(edge_image, 120, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imshow('Edge Image', edge_image)
    cv2.waitKey(0)

    if edge_image is not None:
        
        print ("Deteccion de Rectas con transformada de Hough Comenzada!")
        line_img, lines = busca_rectas(input_img, edge_image, num_rho, num_theta, bin_threshold)
        
        cv2.imshow('Rectas detectadas', line_img)
        cv2.waitKey(0)
        
        line_file = open('rectas_lista.txt', 'w')
        line_file.write('rho, \t theta, \t x1 ,\t y1,  \t x2 ,\t y2 \n')
        for i in range(len(lines)):
            line_file.write(str(lines[i][0]) + ' , ' + str(lines[i][1]) + ' , ' + str(lines[i][2]) + ' , ' + str(lines[i][3]) + ' , ' + str(lines[i][4]) + ' , ' + str(lines[i][5]) + '\n')
        line_file.close()
                
        if line_img is not None:
            cv2.imwrite("rectas_imagen.png", line_img)
    else:
        print ("Error en la imagen de entrada!")
            
    print ("Deteccion de Rectas con transformada de Hough Completada!")



if __name__ == "__main__":
    main()
