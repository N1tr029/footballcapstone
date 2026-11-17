import Foundation
import Vision
import ImageIO

struct Input: Decodable { let png: String }
struct Reading: Encodable { let number: Int; let confidence: Float }

let data = FileHandle.standardInput.readDataToEndOfFile()
var readings: [Reading] = []
if let input = try? JSONDecoder().decode(Input.self, from: data),
   let png = Data(base64Encoded: input.png),
   let source = CGImageSourceCreateWithData(png as CFData, nil),
   let image = CGImageSourceCreateImageAtIndex(source, 0, nil) {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    request.recognitionLanguages = ["en-US"]
    request.usesCPUOnly = true
    if (try? VNImageRequestHandler(cgImage: image, options: [:]).perform([request])) != nil {
        for result in request.results ?? [] {
            for candidate in result.topCandidates(1) {
                let original = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
                var text = original
                if original.contains(where: { $0.isNumber }) {
                    for (letter, digit) in [("I", "1"), ("l", "1"), ("O", "0"), ("Z", "2")] {
                        text = text.replacingOccurrences(of: letter, with: digit)
                    }
                }
                if text.count >= 1 && text.count <= 2 && text.allSatisfy({ $0.isNumber }),
                   let number = Int(text), candidate.confidence >= 0.65 {
                    readings.append(Reading(number: number, confidence: candidate.confidence))
                }
            }
        }
    }
}
if let output = try? JSONEncoder().encode(readings) {
    FileHandle.standardOutput.write(output)
}
