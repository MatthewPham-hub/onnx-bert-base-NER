
using BERTTokenizers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace bert_test
{
    internal class BertTokenizeProgram
    {
        static void Main(string[] args)
        {
            var sentence = "My name is Matthew Pham and I live in Berkeley, California. My email is matthewpham26@gmail.com, my phone number is 5879177910 I work for AHS.";
            // Create Tokenizr and tokenize the sentence.
            var tokenizer = new BertBaseTokenizer();

            // Get the sentence tokens.
            var tokens = tokenizer.Tokenize(sentence);
            // Console.WriteLine(String.Join(", ", tokens));

            // Encode the sentence and pass in the count of the tokens in the sentence.
            var encoded = tokenizer.Encode(tokens.Count(), sentence);

            // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
            var bertInput = new BertInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };
            // Get path to model to create inference session.
            var modelPath = @"C:\Users\Matthew\source\repos\bert-base-NER\bert-base-NER\model.onnx";

            using var runOptions = new RunOptions();
            using var session = new InferenceSession(modelPath);

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                  new long[] { 1, bertInput.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                  new long[] { 1, bertInput.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                  new long[] { 1, bertInput.TypeIds.Length });

            // Create input data for session. Request all outputs in this case.
            var inputs = new Dictionary<string, OrtValue>
  {
      { "input_ids", inputIdsOrtValue },
      { "attention_mask", attMaskOrtValue },
      { "token_type_ids", typeIdsOrtValue }
  };

            // Run session and send the input data in to get inference output. 
            using var output = session.Run(runOptions, inputs, session.OutputNames);
            // Assuming the logits are the first and only output in the output collection.
            var logitsTensor = output.First();

            // Get the tensor data as a read-only span.
            var logitsSpan = logitsTensor.GetTensorDataAsSpan<float>();
            List<string> predictedLabels = new List<string>();
            int numClasses = 9; 
            for (int i = 0; i < logitsSpan.Length; i += numClasses)
            {
                // Find the index of the maximum score for the current token. 1
                int predictedLabelIndex = -1;
                float maxScore = float.MinValue;

                for (int j = 0; j < numClasses; j++)
                {
                    float currentScore = logitsSpan[i + j];
                    if (currentScore > maxScore)
                    {
                        maxScore = currentScore;
                        predictedLabelIndex = j;
                    }
                }

                // Map the predicted index to the corresponding label using id2label function.
                string predictedLabel = id2label(predictedLabelIndex);

                // Add the predicted label to the list.
                predictedLabels.Add(predictedLabel);
                Console.WriteLine($"Token: {tokens[i / numClasses]}, Predicted Label: {predictedLabel}");
            }

            string id2label(int label)
            {
                switch (label)
                {
                    case 0:
                        return "O"; 
                    case 1:
                        return "B-MISC"; 
                    case 2:
                        return "I-MISC"; 
                                        
                    case 3:
                        return "B-PER";
                    case 4:
                        return "I-PER";
                    case 5:
                        return "B-ORG";
                    case 6:
                        return "I-ORG";
                    case 7:
                        return "B-LOC";
                    case 8:
                        return "I-LOC";
                    default:
                        return "UNKNOWN";
                }
            }
        }

    }

}


public struct BertInput
{
    public long[] InputIds { get; set; }
    public long[] AttentionMask { get; set; }
    public long[] TypeIds { get; set; }
}   