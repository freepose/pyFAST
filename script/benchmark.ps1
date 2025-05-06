
# Running on windows

# Set user
$username = "-wzr"

# Set device
$gpu_id = 0

# Set dataset parameters
$datasetName = "ETTh2"
$task = "univariate"
$inputWindowSize = 48
$outputWindowSize = 24

# Set paths
$resultPath = "$env:USERPROFILE/benchmark$username/$datasetName"
$pyfastPath = "$env:USERPROFILE/workspace$username/pyFAST/"
$pythonInterpreter = "C:\Anaconda3\envs\torch2_env\python.exe"

# Set models
$models = @(
    "ar", "rnn", "lstm", "gru", "ed",
    "cnn1d", "cnnrnn", "cnnrnnres", "lstnet",
    "nlinear", "dlinear",
    "transformer", "informer", "autoformer", "fedformer",
    "film", "triformer", "crossformer", "timesnet", "patchtst",
    "staeformer", "itransformer", "timesfm", "timer", "timexer", "timemixer",
    "coat", "tcoat", "codr"
)

# Execute
Write-Output "Step 1. check, if not exist then create: $resultPath"
New-Item -ItemType Directory -Force -Path $resultPath | Out-Null

Write-Output "Step 2. change directory to: $pyfastPath"
if (-not (Test-Path $pyfastPath)) {
    Write-Output "not found $pyfastPath"
    exit 1
}
Set-Location $pyfastPath

Write-Output "Step 3. run models and save logs to: $resultPath"

for ($index = 0; $index -lt $models.Count; $index++) {
    $modelName = $models[$index]
    $indexStr = "{0:D2}" -f ($index + 1)
    $modelLogFilename = "${datasetName}_L${inputWindowSize}_H${outputWindowSize}_${indexStr}_${modelName}.txt"
    
    Write-Output "Validate $modelName, save $modelLogFilename"
    
    & $pythonInterpreter -m example.paper.benchmark `
        --device "cuda:$gpu_id" `
        --dataset_name $datasetName `
        --task $task `
        --input_window_size $inputWindowSize `
        --output_window_size $outputWindowSize `
        --model_name $modelName `
        1> "$resultPath/$modelLogFilename"
}

Write-Output "Done."
