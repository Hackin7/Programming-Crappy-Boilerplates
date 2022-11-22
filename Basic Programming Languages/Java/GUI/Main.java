import javax.swing.*;
import javax.swing.JList;
import javax.swing.JScrollPane;

import java.awt.*;
import java.awt.event.*;

class Main{
    /*public static void main(String args[]){
       JFrame frame = new JFrame("My First GUI");
       frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
       frame.setSize(300,300);
       JButton button = new JButton("Press");
       frame.getContentPane().add(button); // Adds Button to content pane of frame
       
       frame.setVisible(true);
    }*/
    public static void main(String[] args) {
      final JFrame frame = new JFrame();
      frame.setTitle("Title");
      frame.setSize(500, 350);
      frame.setResizable(false);
      frame.setLocationRelativeTo(null);
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

      JButton a = new JButton("button A");
      frame.setContentPane(a);

      frame.setVisible(true); // calling setVisible after content pane has been set to refresh a frame

      a.addActionListener(new ActionListener() {
          @Override
          public void actionPerformed(ActionEvent e) {
              JComponent b = new JLabel("label B");
              frame.setContentPane(b);
              frame.revalidate();
          }
      });
    }
}

/*
https://www.javatpoint.com/java-jtextfield
https://examples.javacodegeeks.com/java-swing-layouts-example/
https://docs.oracle.com/javase/tutorial/uiswing/layout/gridbag.html

https://stackhowto.com/how-to-add-row-dynamically-in-jtable-java/

*/
