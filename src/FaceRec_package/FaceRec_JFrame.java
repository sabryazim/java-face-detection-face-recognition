/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package FaceRec_package;

import java.awt.Image;
import java.awt.Toolkit;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author sabryazim@gmail.com
 */
public class FaceRec_JFrame extends javax.swing.JFrame {
    
    VideoCapture capture;
    Mat frame = new Mat();  //define variable to store the captured frames of the cam
    MatOfByte matOfB = new MatOfByte();  //declare mat of byte variable to encode the captured frames and store them in to display them in the JLabel.

    CascadeClassifier faceCascade = new CascadeClassifier(".\\haarcascade_frontalface_alt_tree.xml");

    LBPHFaceRecognizer recognizer = LBPHFaceRecognizer.create();
    /*
    declare boolean variables to turn on and off the buttons.
    */
    public volatile boolean startCam = false;
    public boolean faceDetect = false;
    public boolean isTrained = false;
    public boolean train = false;
    public boolean savePersonImages = false;
    public boolean predictImages = false;
    
    List<Mat> imgList = new ArrayList<>();
    List<Integer> imgLabels = new ArrayList<>();
    List<String> imgNameList = new ArrayList<>();
    MatOfInt matLabels = new MatOfInt();

    double[] confid = new double[1];
    int[] label = new int[1];


    /**
     * Creates new form FaceRec_JFrame
     */
    class FaceRec_Thread implements Runnable{
        
        @Override
        public void run() {
            synchronized(this){
                while(capture.isOpened()){
                    if(startCam){
                        capture.read(frame);
                        if(faceDetect){
                            Mat grayMat = new Mat();
                            MatOfRect faces = new MatOfRect();
                            Imgproc.cvtColor(frame, grayMat, Imgproc.COLOR_BGR2GRAY);
                            Imgproc.equalizeHist(grayMat, grayMat);
                            faceCascade.detectMultiScale(grayMat, faces);
                            if(faces.toArray().length > 0){
                                for(Rect face : faces.toArray()){
                                    Imgproc.rectangle(frame, new Point(face.x,face.y), new Point(face.x+face.width,face.y+face.height), new Scalar(255,0,0),2);
//                                    Imgproc.putText(frame, "New Face", new Point(face.x-2,face.y-10), 1, 1, new Scalar(0,0,0),2);
                                    Mat faceROI = frame.submat(face);
                                    Imgproc.resize(faceROI, faceROI, new Size(lblRect.getPreferredSize().width,lblRect.getPreferredSize().height));
                                    MatOfByte matOfB_faceROI = new MatOfByte();
                                    Imgcodecs.imencode(".jpg",faceROI, matOfB_faceROI);
                                    Image Img_faceROI = Toolkit.getDefaultToolkit().createImage(matOfB_faceROI.toArray());
                                    lblRect.setIcon(new ImageIcon(Img_faceROI));
                                    if(savePersonImages){
                                        btnSavePersonImage.setEnabled(false);
                                        String cur_path = Paths.get("").toAbsolutePath().toString()+"\\images";
                                        File DirOfImages = new File(cur_path);
                                        if(!DirOfImages.exists()){
                                            DirOfImages.mkdir();
                                        }
                                        Thread Th_copy = new Thread(new Runnable() {
                                            @Override
                                            public void run() {
                                                try {
                                                    for(int i = 0 ; i < 20 ;i++){
                                                        MatOfByte matOfB1 = new MatOfByte();
                                                        Mat mat1 = new Mat();
                                                        Imgproc.cvtColor(faceROI, mat1, Imgproc.COLOR_RGB2GRAY);
                                                        Imgproc.equalizeHist(mat1, mat1);
                                                        
                                                        Imgproc.resize(mat1, mat1, new Size(200,200));
                                                        Imgcodecs.imencode(".jpg", mat1, matOfB1);
                                                        String personName = txtPersonName.getText().isEmpty()?"file":txtPersonName.getText();
                                                        Path ImgPath = Paths.get(""+"images\\"+personName+System.nanoTime()+".jpg");
                                                        Files.copy(new ByteArrayInputStream(matOfB1.toArray()), ImgPath, StandardCopyOption.REPLACE_EXISTING);
                                                        Thread.sleep(50);
                                                    }
                                                    
                                                } catch (Exception ex) {
                                                    Logger.getLogger(FaceRec_JFrame.class.getName()).log(Level.SEVERE, null, ex);
                                                }
                                                savePersonImages = false;
                                                btnSavePersonImage.setEnabled(true);
                                            }
                                        });
                                        Th_copy.start();
                                    }

                                    if(predictImages){
                                        label[0] = 0;
                                        confid[0] = 0.0;
                                        recognizer.clear();
                                        try{
                                            Mat grayFaceRes = new Mat();
                                            Imgproc.cvtColor(faceROI, grayFaceRes, Imgproc.COLOR_RGB2GRAY);
                                            Imgproc.resize(grayFaceRes, grayFaceRes, new Size(200,200));
                                            Imgproc.equalizeHist(grayFaceRes, grayFaceRes);
                                            recognizer.predict(grayFaceRes, label, confid);
                                            int result = recognizer.predict_label(grayFaceRes);
                                            System.out.println("this is result : "+result);
                                            System.out.println("this is label : "+label[0]);
                                            System.out.println("this is confid : "+confid[0]);
                                            if(result != -1){
                                                Imgproc.putText(frame, imgNameList.get(result), new Point(face.x-2,face.y-10), 1, 1, new Scalar(250,250,250),2);
                                                MatOfByte matbb = new MatOfByte();
                                                Mat mmat = imgList.get(result);
                                                Imgproc.resize(mmat, mmat, new Size(141,123));
                                                Imgcodecs.imencode(".jpg", mmat, matbb);
                                                Image imgPredicted = Toolkit.getDefaultToolkit().createImage(matbb.toArray());
                                                lblPredicted.setIcon(new ImageIcon(imgPredicted));
                                            }
//                                            predictImages = false;
                                        }catch(Exception e){
                                            JOptionPane.showMessageDialog(null, e.toString(), "ERROR",JOptionPane.WARNING_MESSAGE);
//                                            predictImages = false;
                                        }
                                    }
                                }
                            }
                        }
                        Imgproc.resize(frame, frame, new Size(lblCapture.getPreferredSize().width,lblCapture.getPreferredSize().height));
                        Imgcodecs.imencode(".jpg", frame, matOfB);
                        Image img = Toolkit.getDefaultToolkit().createImage(matOfB.toArray());
                        lblCapture.setIcon(new ImageIcon(img));
                    }
                }
            }

        }
    }
    
    public void trainImages(){
        imgList.clear();
        imgLabels.clear();
        imgNameList.clear();
        File imgFolder = new File(Paths.get("").toAbsolutePath().toString()+"\\images");
        File[] listOfImages = imgFolder.listFiles();
        int[] intLabels;
        if(listOfImages != null){
            intLabels = new int[listOfImages.length];
            System.out.println(imgFolder);
            for (File listOfImage : listOfImages) {
                if (listOfImage.getName().split("\\.")[1].contains("jpg")) {
                    System.out.println(listOfImage.getName().split("\\.")[0]);
                }
            }
            int count = 0;
            if(listOfImages.length > 0){
                for(File listOfImage : listOfImages){
                        System.out.println(listOfImage.getAbsoluteFile().toString());
                        Mat mat = Imgcodecs.imread(listOfImage.getAbsoluteFile().toString(),Imgcodecs.IMREAD_GRAYSCALE);
    //                    Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY);
    //                    Imgproc.resize(grayMat, grayMat, new Size(200,200));
    //                    Imgproc.equalizeHist(grayMat, grayMat);
                        System.out.println(mat);
                        imgList.add(count, mat);
                        intLabels[count] = count;
                        System.out.println("this is  "+imgList.get(count));
                        imgNameList.add(listOfImage.getName().split("\\.")[0]);
                        count++;
                }
            }
            System.out.println(imgList.size());
            if(imgList.size() > 0){
                matLabels.fromArray(intLabels);
                recognizer.train(imgList, matLabels);
    //            isTrained = true;
    //            return true;
            }else{
                System.out.println("there is not images to train");
    //            isTrained = false;
    //            return false;
            }
        }else{
            JOptionPane.showMessageDialog(null,"there are not saved images in folder, CREATE IT !!","ERROR", WIDTH);
        }

    }
    
    public FaceRec_JFrame() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        txtPersonName = new javax.swing.JTextField();
        jLabel1 = new javax.swing.JLabel();
        lblCapture = new javax.swing.JLabel();
        lblRect = new javax.swing.JLabel();
        lblTrained = new javax.swing.JLabel();
        lblPredicted = new javax.swing.JLabel();
        btnCapture = new javax.swing.JButton();
        btnTrain = new javax.swing.JButton();
        btnPredict = new javax.swing.JButton();
        btnSavePersonImage = new javax.swing.JButton();
        btnDetect = new javax.swing.JButton();
        btnStopCam = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        txtPersonName.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N

        jLabel1.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        jLabel1.setText("Name of Person : ");

        lblCapture.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        lblCapture.setPreferredSize(new java.awt.Dimension(516, 395));

        lblRect.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        lblRect.setPreferredSize(new java.awt.Dimension(141, 123));

        lblTrained.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        lblTrained.setPreferredSize(new java.awt.Dimension(141, 123));

        lblPredicted.setBorder(javax.swing.BorderFactory.createEtchedBorder());
        lblPredicted.setPreferredSize(new java.awt.Dimension(141, 123));

        btnCapture.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnCapture.setText("Start webcam");
        btnCapture.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnCaptureActionPerformed(evt);
            }
        });

        btnTrain.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnTrain.setText("Train");
        btnTrain.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                btnTrainActionPerformed(evt);
            }
        });

        btnPredict.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnPredict.setText("Predict");

        btnSavePersonImage.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnSavePersonImage.setText("save person image");

        btnDetect.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnDetect.setText("Detect Face");

        btnStopCam.setFont(new java.awt.Font("Tahoma", 1, 12)); // NOI18N
        btnStopCam.setText("Stop cam");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblCapture, javax.swing.GroupLayout.PREFERRED_SIZE, 516, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED, 32, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(lblTrained, javax.swing.GroupLayout.PREFERRED_SIZE, 141, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(lblPredicted, javax.swing.GroupLayout.PREFERRED_SIZE, 141, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(lblRect, javax.swing.GroupLayout.PREFERRED_SIZE, 141, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(btnCapture, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                        .addGap(18, 18, 18)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(txtPersonName, javax.swing.GroupLayout.PREFERRED_SIZE, 255, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(btnDetect)
                                .addGap(18, 18, 18)
                                .addComponent(btnSavePersonImage)
                                .addGap(18, 18, 18)
                                .addComponent(btnTrain)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(btnPredict)
                                .addGap(18, 18, 18)
                                .addComponent(btnStopCam, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))))
                .addGap(18, 18, 18))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(lblRect, javax.swing.GroupLayout.PREFERRED_SIZE, 123, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                        .addComponent(lblTrained, javax.swing.GroupLayout.PREFERRED_SIZE, 123, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(lblPredicted, javax.swing.GroupLayout.PREFERRED_SIZE, 123, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(lblCapture, javax.swing.GroupLayout.PREFERRED_SIZE, 395, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(txtPersonName)
                    .addComponent(jLabel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(btnCapture, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnTrain, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnPredict, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnSavePersonImage, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnDetect, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(btnStopCam, javax.swing.GroupLayout.PREFERRED_SIZE, 43, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(20, 20, 20))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void btnCaptureActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnCaptureActionPerformed
        capture = new VideoCapture(0);
        FaceRec_Thread runn = new FaceRec_Thread();
        startCam = true;
        Thread th = new Thread(runn);
        th.start();
        btnCapture.setEnabled(false);
        
        ActionListener alStop = (ActionEvent ae) -> {
            th.interrupt();
            startCam = false;
            capture.release();
            lblCapture.setIcon(null);
            btnCapture.setEnabled(true);
        };
        btnStopCam.addActionListener(alStop);
        
        ActionListener alDetectFace = (ActionEvent ae) -> {
            faceDetect = true;
        };
        btnDetect.addActionListener(alDetectFace);
        
        ActionListener alSaveImages = (ActionEvent ae) -> {
            savePersonImages = true;
        };
        btnSavePersonImage.addActionListener(alSaveImages);
        
//        ActionListener alTrain = (ActionEvent ae) -> {
////            train = true;
//            trainImages();
//        };
//        btnTrain.addActionListener(alTrain);
        
        ActionListener alPredictImages = (ActionEvent ae) -> {
            predictImages = true;
        };
        btnPredict.addActionListener(alPredictImages);
    }//GEN-LAST:event_btnCaptureActionPerformed

    private void btnTrainActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_btnTrainActionPerformed
        // TODO add your handling code here:
        trainImages();
    }//GEN-LAST:event_btnTrainActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(FaceRec_JFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(FaceRec_JFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(FaceRec_JFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(FaceRec_JFrame.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
      
        
        /* Create and display the form */
        java.awt.EventQueue.invokeLater(() -> {
            new FaceRec_JFrame().setVisible(true);
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton btnCapture;
    private javax.swing.JButton btnDetect;
    private javax.swing.JButton btnPredict;
    private javax.swing.JButton btnSavePersonImage;
    private javax.swing.JButton btnStopCam;
    private javax.swing.JButton btnTrain;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel lblCapture;
    private javax.swing.JLabel lblPredicted;
    private javax.swing.JLabel lblRect;
    private javax.swing.JLabel lblTrained;
    private javax.swing.JTextField txtPersonName;
    // End of variables declaration//GEN-END:variables
}
