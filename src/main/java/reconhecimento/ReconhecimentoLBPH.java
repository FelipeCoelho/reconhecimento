package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.FONT_HERSHEY_PLAIN;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.putText;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.awt.event.KeyEvent;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class ReconhecimentoLBPH {

	public static void main(String[] args) throws Exception {
		KeyEvent tecla = null;
		
		String[] pessoas = {"","Felipe","Rodrigo","Jaqueline"}; 
		OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
		OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
		camera.start();

		CascadeClassifier detectorFace = new CascadeClassifier(
				"src\\main\\java\\recursos\\haarcascade-frontalface-alt.xml");

		FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
		reconhecedor.read("src\\\\recursos\\\\classificadorLBPH.yml");
//		reconhecedor.setThreshold(2000);
		
		CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
		Frame frameCapturado = null;
		Mat imagemColorida = new Mat();

		while ((frameCapturado = camera.grab()) != null) {
			imagemColorida = convertMat.convert(frameCapturado);
			Mat imagemCinza = new Mat();
			cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
			RectVector facesDetectadas = new RectVector();
			detectorFace.detectMultiScale(imagemCinza, facesDetectadas, 1.1, 1, 0, new Size(150, 150),
					new Size(500, 500));
			for (int i = 0; i < facesDetectadas.size(); i++) {
				Rect dadosFace = facesDetectadas.get(i);
				rectangle(imagemColorida, dadosFace, new Scalar(0, 0, 255, 0));
				Mat faceCapturada = new Mat(imagemCinza, dadosFace);
				resize(faceCapturada, faceCapturada, new Size(160, 160));
				
				IntPointer rotulo =  new IntPointer(1);
				DoublePointer confianca = new DoublePointer(1);
				
				reconhecedor.predict(faceCapturada, rotulo, confianca);
				int predicao = rotulo.get(0);
				
				String nome;
				
				if(predicao == -1) {
					nome = "Desconhecido";
				} else {
					nome = pessoas[predicao] + " - " + confianca.get(0);
				}
				
				int x  = Math.max(dadosFace.tl().x() - 10, 0);
				int y  = Math.max(dadosFace.tl().y() - 10, 0);
				
				putText(imagemColorida, nome ,new Point(x,y), FONT_HERSHEY_PLAIN,1.4, new Scalar(0,255,0,0));
			}

			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}
		}

		cFrame.dispose();
		camera.stop();

	}
}
