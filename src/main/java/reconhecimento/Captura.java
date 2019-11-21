package reconhecimento;

import static org.bytedeco.opencv.global.opencv_imgproc.CV_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

import java.awt.event.KeyEvent;
import java.util.Scanner;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class Captura {

	public static void main(String[] args) throws Exception {
		KeyEvent tecla = null;
		OpenCVFrameConverter.ToMat convertMat = new OpenCVFrameConverter.ToMat();
		OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
		camera.start();
		
		CascadeClassifier detectorFace  = new CascadeClassifier("src\\main\\java\\recursos\\haarcascade-frontalface-alt.xml");
		
		CanvasFrame cFrame = new CanvasFrame("Preview", CanvasFrame.getDefaultGamma() / camera.getGamma());
		Frame frameCapturado = null;
		Mat imagemColorida = new Mat();
		int numeroAmostras = 25;
		int amostra = 1;
		
		System.out.println("Digite seu Id: ");
		
		Scanner cadastro  = new Scanner(System.in);
		int idPessoa = cadastro.nextInt();
		while((frameCapturado = camera.grab()) != null) {
			imagemColorida = convertMat.convert(frameCapturado);
			Mat imagemCinza  = new Mat();

			cvtColor(imagemColorida, imagemCinza, CV_BGR2GRAY);
			
			RectVector facesDetectadas = new RectVector();
			detectorFace.detectMultiScale(imagemCinza, facesDetectadas,1.1 , 1, 0, new Size(150,150), new Size(500,500));
			
			for (int i = 0; i < facesDetectadas.size(); i++) {
				Rect dadosFace = facesDetectadas.get(0);
				rectangle(imagemColorida, dadosFace, new Scalar(0,0,255,0));
			}
			
			if (cFrame.isVisible()) {
				cFrame.showImage(frameCapturado);
			}
		}
		
		cFrame.dispose();
		camera.stop();
		
	}
}
