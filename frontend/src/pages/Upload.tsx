import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileUploader, FileUploaderContent, FileUploaderItem, FileInput } from "@/components/ui/fileinput";
import { Paperclip } from "lucide-react";
import { Button } from "@/components/ui/button";
import { uploadImage } from "@/lib/api";

const Upload = () => {
    const [files, setFiles] = useState<File[] | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const navigate = useNavigate();

    const handleFileSelect = (file: File[] | null) => {
        setFiles(file ?? []);
        setError(null); // Clear any previous errors
    };

    const handleUpload = async () => {
        if (!files || files.length === 0) {
            setError("Please select an image before analyzing.");
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const data = await uploadImage(files[0]);
            console.log("Prediction Response:", data);

            // Navigate to Results page with the uploaded image and API response
            navigate("/results", {
                state: {
                    image: files[0],
                    prediction: data, // API response
                },
            });
        } catch (err) {
            console.error("Error uploading file:", err);
            setError("There was an error processing your request. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const dropZoneConfig = {
        maxFiles: 1,
        maxSize: 1024 * 1024 * 4, // 4MB
        multiple: false,
    };

    return (
        //bg-gray-100 dark:bg-gray-900
        <section className="min-h-screen flex flex-col items-center justify-center bg-gray-100 dark:bg-gray-900 p-6">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg space-y-6">
                <h1 className="text-3xl font-bold text-center text-gray-900 dark:text-white">
                    Upload Ultrasound Scan
                </h1>

                <FileUploader
                    value={files}
                    onValueChange={handleFileSelect}
                    dropzoneOptions={dropZoneConfig}
                    className="relative bg-background rounded-lg p-2"
                >
                    <FileInput className="p-6 outline-dashed outline-1 outline-black dark:outline-gray-400">
                        <div className="flex items-center justify-center flex-col pt-3 pb-4 w-full">
                            <Paperclip className="h-8 w-8 text-gray-500 dark:text-gray-400" />
                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                Click here to upload or drag and drop
                            </p>
                        </div>
                    </FileInput>
                    <FileUploaderContent>
                        {files &&
                            files.map((file, i) => (
                                <FileUploaderItem key={i} index={i}>
                                    <Paperclip className="h-4 w-4 stroke-current" />
                                    <span>{file.name}</span>
                                </FileUploaderItem>
                            ))}
                    </FileUploaderContent>
                </FileUploader>

                {error && <p className="text-red-500 text-center">{error}</p>}

                <Button onClick={handleUpload} disabled={!files || loading} className="w-full">
                    {loading ? "Analyzing..." : "Analyze"}
                </Button>
            </div>
        </section>
    );
};

export default Upload;
