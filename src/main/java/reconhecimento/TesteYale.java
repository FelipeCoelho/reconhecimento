package reconhecimento;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.File;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

public class TesteYale {
    public static void main(String[] args) {
        int totalAcertos = 0;
        double percentualAcerto = 0;
        double totalConfianca = 0;
        
//        FaceRecognizer reconhecedor  = EigenFaceRecognizer.create();
//		FaceRecognizer reconhecedor = FisherFaceRecognizer.create();
		FaceRecognizer reconhecedor = LBPHFaceRecognizer.create();
        
        //FaceRecognizer reconhecedor = createEigenFaceRecognizer();
        //FaceRecognizer reconhecedor = createFisherFaceRecognizer();
//        FaceRecognizer reconhecedor = createLBPHFaceRecognizer();

        //reconhecedor.load("src\\recursos\\classificadorEigenfacesYale.yml");
        //reconhecedor.load("src\\recursos\\classificadorFisherfacesYale.yml");
        reconhecedor.read("src\\recursos\\classificadorLBPHYale.yml");

        File diretorio = new File("src\\yalefaces\\teste");
        File[] arquivos = diretorio.listFiles();
        
        for (File imagem : arquivos) {           
            Mat foto = imread(imagem.getAbsolutePath(), IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().substring(7, 9));          
            resize(foto, foto, new Size(160, 160));

            IntPointer rotulo = new IntPointer(1);
            DoublePointer confianca = new DoublePointer(1);
            reconhecedor.predict(foto, rotulo, confianca);
            int predicao = rotulo.get(0);
            System.out.println(classe + " foi reconhecido como " + predicao + " - " + confianca.get(0));
            if (classe == predicao) {
                totalAcertos++;
                totalConfianca += confianca.get(0);
            }
        }
        
        percentualAcerto = (totalAcertos / 30.0) * 100;
        totalConfianca = totalConfianca / totalAcertos;
        System.out.println("Percentual de acerto: " + percentualAcerto);
        System.out.println("Total confiança: " + totalConfianca);
    }
}