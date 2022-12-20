//
//  ViewController.swift
//  MLImgDetection
//
//  Created by cristopher cruz on 12/20/22.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    // Text-field that will go under the imageView
    @IBOutlet weak var imgDesc: UITextView!
    
    // Image view object
    @IBOutlet weak var imageViewObject: UIImageView!
    
    // Image picker
    var imagePicker:UIImagePickerController!
    override func viewDidLoad() {
        super.viewDidLoad()
        imagePicker=UIImagePickerController()
        imagePicker.delegate = self
        imagePicker.sourceType = .camera
    }
    
    // Button action to take a picture
    @IBAction func takePictureButton(_ sender: Any) {
        present(imagePicker, animated: true)
        
    }
    
    // cast image as UIImage
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        imageViewObject.image=info[UIImagePickerController.InfoKey.originalImage] as? UIImage
        imagePicker.dismiss(animated: true, completion: nil)
        pictureIdentifyML(image: (info[UIImagePickerController.InfoKey.originalImage] as? UIImage)!)
    }
    
    // Function that takes image as input to identify
    func pictureIdentifyML(image:UIImage){
        guard let mlModel = try? Resnet50(configuration: .init()).model,
              let model = try? VNCoreMLModel(for: mlModel) else {
            print("cannot load ML model!")
            return
        }
        
        // Sending request to CoreML
        let request = VNCoreMLRequest(model:model){
            [ weak self] request, error in
            
            // get results, If unable to get data, print error
            guard let results = request.results as? [VNClassificationObservation],
                  let firstResult = results.first else{
                fatalError("cannot get result from VNCoreMLRequest")
            }
            
            // If able to get data, show results with identifier and confidence of results
            DispatchQueue.main.async {
                self?.imgDesc.text = "Object : \((firstResult.identifier)) \n Confidence = \( Int(firstResult.confidence * 100))% "
                
            }
        }
        
        // ----- * send image to start processing * -----
        
        // convert image to CIImage
        guard let ciImage = CIImage(image: image) else {
            fatalError("cannot convert to CIimage  ")
        }
        
        // define handler
        let imageHandler = VNImageRequestHandler(ciImage:ciImage)
        
        // perform request and notify me if error
        DispatchQueue.global(qos: .userInteractive).async {
            do{
                try imageHandler.perform([request])
            }catch{
                print("Error \(error)")
            }
        }
        
        
    }
}
