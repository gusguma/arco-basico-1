///////////////////////////////////////////////////////////////////////////
/// PROGRAMACIÓN EN CUDA C/C++
/// Práctica:	BASICO 1 : Memoria Global
/// Autor:		Gustavo Gutierrez Martin
/// Fecha:		Septiembre 2022
///////////////////////////////////////////////////////////////////////////

/// dependencias ///
#include <cstdio>
#include <ctime>

/// constantes ///
#define MB (1<<20) /// MiB = 2^20
#define N 16 /// Tamaño del array de datos

/// definición de funciones ///

/// muestra por consola que no se ha encontrado un dispositivo CUDA
int getErrorDevice();

/// muestra los datos de los dispositivos CUDA encontrados
int getDataDevice(int deviceCount);

/// numero de CUDA cores
int getCudaCores(cudaDeviceProp deviceProperties);

/// muestra por pantalla las propiedades del dispositivo CUDA
int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties);

/// función que muestra por pantalla la salida del programa
int getAppOutput();

/// muestra por pantalla los datos del host
int printHostData(float *hst_A, float *hst_B);

/// inicializa el array del host
int loadHostData(float *hst_A, float *hst_B);

/// transfiere los datos
int dataTransfer(float *hst_A, float *hst_B,float *dev_A, float *dev_B );

/// función principal de la aplicación
int main() {
    /// declaración de variables
    int deviceCount;
    float *hst_A,*hst_B,*dev_A,*dev_B;

    /// reserva del espacio de memoria en el host
    hst_A = (float*)malloc( N * sizeof(float) );
    hst_B = (float*)malloc( N * sizeof(float) );
    /// reserva del espacio de memoria en el device
    cudaMalloc( (void**)&dev_A, N * sizeof(float) );
    cudaMalloc( (void**)&dev_B, N * sizeof(float) );

    /// cargamos los datos iniciales en el host
    loadHostData(hst_A, hst_B);

    /// transferimos los datos
    dataTransfer(hst_A, hst_B, dev_A, dev_B);

    /// buscando dispositivos
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        /// mostramos el error si no se encuentra un dispositivo
        return getErrorDevice();
    } else {
        /// mostramos los datos de los dispositivos CUDA encontrados
        getDataDevice(deviceCount);
    }
    /// mostramos por pantalla los datos del host
    printHostData(hst_A, hst_B);

    /// liberamos los recursos del device
    cudaFree(dev_A);
    cudaFree(dev_B);

    /// mostramos el final del programa
    getAppOutput();
    return 0;
}

int getErrorDevice() {
    printf("¡No se ha encontrado un dispositivo CUDA!\n");
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 1;
}

int getDataDevice(int deviceCount) {
    printf("Se han encontrado %d dispositivos CUDA:\n", deviceCount);
    for (int deviceID = 0; deviceID < deviceCount; deviceID++) {
        ///obtenemos las propiedades del dispositivo CUDA
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, deviceID);
        getDeviceProperties(deviceID, getCudaCores(deviceProp), deviceProp);
    }
    return 0;
}

int getCudaCores(cudaDeviceProp deviceProperties) {
    int cudaCores = 0;
    int major = deviceProperties.major;
    if (major == 1) {
        /// TESLA
        cudaCores = 8;
    } else if (major == 2) {
        /// FERMI
        if (deviceProperties.minor == 0) {
            cudaCores = 32;
        } else {
            cudaCores = 48;
        }
    } else if (major == 3) {
        /// KEPLER
        cudaCores = 192;
    } else if (major == 5) {
        /// MAXWELL
        cudaCores = 128;
    } else if (major == 6 || major == 7 || major == 8) {
        /// PASCAL, VOLTA (7.0), TURING (7.5), AMPERE
        cudaCores = 64;
    } else {
        /// ARQUITECTURA DESCONOCIDA
        cudaCores = 0;
        printf("¡Dispositivo desconocido!\n");
    }
    return cudaCores;
}

int getDeviceProperties(int deviceId, int cudaCores, cudaDeviceProp cudaProperties) {
    int SM = cudaProperties.multiProcessorCount;
    printf("***************************************************\n");
    printf("DEVICE %d: %s\n", deviceId, cudaProperties.name);
    printf("***************************************************\n");
    printf("- Capacidad de Computo            \t: %d.%d\n", cudaProperties.major, cudaProperties.minor);
    printf("- No. de MultiProcesadores        \t: %d \n", SM);
    printf("- No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
    printf("- Memoria Global (total)          \t: %zu MiB\n", cudaProperties.totalGlobalMem / MB);
    printf("***************************************************\n");
    return 0;
}

int getAppOutput() {
    /// salida del programa
    time_t fecha;
    time(&fecha);
    printf("***************************************************\n");
    printf("Programa ejecutado el: %s", ctime(&fecha));
    printf("***************************************************\n");
    /// capturamos un INTRO para que no se cierre la consola de MSVS
    printf("<pulsa [INTRO] para finalizar>");
    getchar();
    return 0;
}

int printHostData(float *hst_A, float *hst_B) {
    printf("ENTRADA:\n");
    for (int i=0; i<N; i++)  {
        printf("HST_A[%i] = %.2f\n", i, hst_A[i]);
    }
    printf("\n");
    printf("SALIDA:\n");
    for (int i=0; i<N; i++)  {
        printf("HST_B[%i] = %.2f\n", i, hst_B[i]);
    }
    printf("\n");
    return 0;
}

int loadHostData(float *hst_A, float *hst_B) {
    srand ( (int)time(nullptr) );
    for (int i=0; i<N; i++)  {
        /// inicializamos hst_A con numeros aleatorios entre 0 y 1
        hst_A[i] = (float) rand() / RAND_MAX;
        /// inicializamos hst_B con ceros
        hst_B[i] = 0;
    }
    return 0;
}

int dataTransfer(float *hst_A, float *hst_B,float *dev_A, float *dev_B ) {
    /// transfiere datos de hst_A a dev_A
    cudaMemcpy(dev_A,hst_A,N * sizeof(float),cudaMemcpyHostToDevice);
    /// transfiere datos de dev_A a dev_B
    cudaMemcpy(dev_B,dev_A,N * sizeof(float),cudaMemcpyDeviceToDevice);
    /// transfiere datos de dev_B a hst_B
    cudaMemcpy(hst_B,dev_B,N * sizeof(float),cudaMemcpyDeviceToHost);
    return 0;
}
