import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class OpencvJavaTest {
	
	static {
		System.loadLibrary("opencv_java310");
	}
	
	
	public static Mat wbImage(Mat imagem) {
		Mat dst = new Mat();
		Imgproc.cvtColor(imagem, dst, Imgproc.COLOR_BGR2GRAY);
		Imgproc.adaptiveThreshold(dst, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 30);
		return dst;
		
	}
	
	
	public static Mat cropImage(Mat imagemOriginal, MatOfPoint2f border) {
		
		List<Point> lista = border.toList();
		
		Collections.sort(lista, (p, p1) -> {
			return new Double(p.y).intValue() - new Double(p1.y).intValue();
		});
		
		
		Point tl = null;
		Point tr = null;
		Point br = null;
		Point bl = null;
		
		if(lista.get(0).x < lista.get(1).x) {
			
			bl = lista.get(0);
			br = lista.get(1);
			
		} else {
			bl = lista.get(1);
			br = lista.get(0);
		}
		
		if(lista.get(2).x < lista.get(3).x) {
			tl = lista.get(2);
			tr = lista.get(3);
		} else {
			tl = lista.get(3);
			tr = lista.get(2);
		}
		
	    
	    Mat src_mat=new Mat(4,1,CvType.CV_32FC2);
	    Mat dst_mat=new Mat(4,1,CvType.CV_32FC2);
	    
	    
	    double widthA = Math.sqrt(Math.pow((br.x - bl.x), 2) + Math.pow((br.y - bl.y), 2));
		double widthB = Math.sqrt(Math.pow((tr.x - tl.x), 2) + Math.pow((tr.y - tl.y), 2));
		double maxWidth = Math.max(widthA, widthB);

		
		double heightA = Math.sqrt(Math.pow((tr.x - br.x), 2) + Math.pow((tr.y - br.y), 2));
		double heightB = Math.sqrt(Math.pow((tl.x - bl.x), 2) + Math.pow((tl.y - bl.y), 2));
		double maxHeight = Math.max(heightA, heightB);
		

	    src_mat.put(0, 0,
	    		bl.x, bl.y,
	    		br.x, br.y,
	    		tr.x, tr.y,
	    		tl.x, tl.y
	    		);
	    
	    dst_mat.put(0, 0,
	    			0, 0,
	    			maxWidth - 1, 0,
	    			maxWidth - 1, maxHeight - 1,
	    			0.0, maxHeight - 1);
	    
	    Mat perspectiveTransform = Imgproc.getPerspectiveTransform(src_mat, dst_mat);
	    Mat dst = imagemOriginal.clone();
	    Imgproc.warpPerspective(imagemOriginal, dst, perspectiveTransform, new Size(maxWidth, maxHeight));
	    
	    return dst;
		
	}
	
	
	
	
	public static MatOfPoint2f findBorders(Mat image, boolean draw) {
		
		MatOfPoint2f response = new MatOfPoint2f();
		
		Mat imgGrayscale = new Mat();
		Mat imgBlurred = new Mat();
		Mat imgCanny = new Mat();
		
		Imgproc.cvtColor(image, imgGrayscale, Imgproc.COLOR_BGR2GRAY);
		Imgproc.GaussianBlur(imgGrayscale, imgBlurred, new Size(5, 5), 1.8);
		Imgproc.Canny(imgBlurred, imgCanny,	50, 100);
		
		List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(imgCanny, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
		
		Collections.sort(contours, (list, list1) -> {
			
			MatOfPoint2f resp = new MatOfPoint2f();
			MatOfPoint2f resp1 = new MatOfPoint2f();
			
			double peri = Imgproc.arcLength(new MatOfPoint2f(list.toArray()), true);
			Imgproc.approxPolyDP(new MatOfPoint2f(list.toArray()), resp, 0.02 * peri, true);
			
			
			peri = Imgproc.arcLength(new MatOfPoint2f(list1.toArray()), true);
			Imgproc.approxPolyDP(new MatOfPoint2f(list1.toArray()), resp1, 0.02 * peri, true);
			
			return new Double(Imgproc.contourArea(resp) - Imgproc.contourArea(resp1)).intValue();
			
		});
		
		Collections.reverse(contours);
		
		
		
		for(int i = 0; i < contours.size(); i++) {
			
			double peri = Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true);
			Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), response, 0.02 * peri, true);
			
			if(response.toArray().length == 4) {
				
				System.out.println(Imgproc.contourArea(response));
				
				if(draw) {
					Imgproc.drawContours(image, contours, i, new Scalar(0, 0, 255), 3);
				}
				break;
			}
		}
		
		return response;
		
	}
	
	
	
	public static void main(String args[]) {
		
		Mat image = Imgcodecs.imread("c:\\cropTest\\receipt_real.jpg", 1);
		MatOfPoint2f points = findBorders(image, true);
		
		
		//Imshow.show(image);
		
		Mat imagemTratada = wbImage(cropImage(image, points));
		
		
		
		
		Imgcodecs.imwrite("C:\\cropTest\\saida.jpg", imagemTratada);
		
		
	}

}
