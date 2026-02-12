#define _GNU_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static FILE *logf = NULL;
static int world_rank_cached = -1;
static int initialized = 0;

static void init_log(void) __attribute__((constructor));
static void close_log(void) __attribute__((destructor));

static void init_log(void) {
    const char *fname = getenv("MPI_MSG_LOG_FILE");
    if (!fname) {
        fname = "mpi_msg_sizes.log";
    }

    logf = fopen(fname, "w");
    if (!logf) {
        logf = stderr;
        fprintf(stderr,
                "mpi_msg_logger: could not open %s, logging to stderr\n",
                fname);
    }
}

static void close_log(void) {
    if (logf && logf != stderr) {
        fclose(logf);
    }
}

static int is_mpi_initialized(void) {
    int flag = 0;
    PMPI_Initialized(&flag);
    return flag;
}

static int get_world_rank(void) {
    if (!is_mpi_initialized()) {
        return -1; /* Not safe to call Comm_rank yet */
    }

    if (world_rank_cached < 0) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank_cached);
    }
    return world_rank_cached;
}

static size_t compute_bytes(int count, MPI_Datatype datatype) {
    int typesize = 0;
    PMPI_Type_size(datatype, &typesize);
    return (size_t)count * (size_t)typesize;
}

static void log_msg(const char *fmt, ...) {
    if (!logf) return;

    va_list ap;
    va_start(ap, fmt);
    vfprintf(logf, fmt, ap);
    va_end(ap);
    fflush(logf);
}


int MPI_Init(int *argc, char ***argv)
{
    int rc = PMPI_Init(argc, argv);
    initialized = 1;
    int wrank = get_world_rank();
    double t = PMPI_Wtime();
    if (logf) {
        fprintf(logf, "[%f] MPI_Init: world_rank=%d\n", t, wrank);
        fflush(logf);
    }
    return rc;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided)
{
    int rc = PMPI_Init_thread(argc, argv, required, provided);
    initialized = 1;
    int wrank = get_world_rank();
    double t = PMPI_Wtime();
    if (logf) {
        fprintf(logf, "[%f] MPI_Init_thread: world_rank=%d required=%d provided=%d\n",
                t, wrank, required, *provided);
        fflush(logf);
    }
    return rc;
}

int MPI_Finalize(void)
{
    int wrank = get_world_rank();
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;
    if (logf) {
        fprintf(logf, "[%f] MPI_Finalize: world_rank=%d\n", t, wrank);
        fflush(logf);
    }
    initialized = 0;
    return PMPI_Finalize();
}

int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
             int dest, int tag, MPI_Comm comm)
{
    size_t bytes = compute_bytes(count, datatype);
    int rank = -1;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Send: world_rank=%d comm_rank=%d -> dest=%d tag=%d bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, dest, tag, bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status)
{
    size_t bytes = compute_bytes(count, datatype);
    int rank = -1;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Recv: world_rank=%d comm_rank=%d <- source=%d tag=%d bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, source, tag, bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Recv(buf, count, datatype, source, tag, comm, status);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                 int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype,
                 int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status)
{
    size_t send_bytes = compute_bytes(sendcount, sendtype);
    size_t recv_bytes = compute_bytes(recvcount, recvtype);
    int rank = -1;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Sendrecv: world_rank=%d comm_rank=%d "
                "send-> dest=%d tag=%d send_bytes=%zu ; "
                "recv<- source=%d tag=%d recv_bytes=%zu comm=%p\n",
                t, get_world_rank(), rank,
                dest, sendtag, send_bytes,
                source, recvtag, recv_bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Sendrecv(sendbuf, sendcount, sendtype,
                         dest, sendtag,
                         recvbuf, recvcount, recvtype,
                         source, recvtag,
                         comm, status);
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm)
{
    size_t bytes = compute_bytes(count, datatype);
    int rank = -1, size = 0;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Bcast: world_rank=%d comm_rank=%d comm_size=%d "
                "root=%d bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, size, root, bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Bcast(buffer, count, datatype, root, comm);
}

int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  MPI_Comm comm)
{
    size_t send_bytes = compute_bytes(sendcount, sendtype);
    size_t recv_bytes = compute_bytes(recvcount, recvtype);
    int rank = -1, size = 0;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Allgather: world_rank=%d comm_rank=%d comm_size=%d "
                "send_bytes=%zu recv_bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, size,
                send_bytes, recv_bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Allgather(sendbuf, sendcount, sendtype,
                          recvbuf, recvcount, recvtype, comm);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf,
                  int count, MPI_Datatype datatype,
                  MPI_Op op, MPI_Comm comm)
{
    size_t bytes = compute_bytes(count, datatype);
    int rank = -1, size = 0;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Allreduce: world_rank=%d comm_rank=%d comm_size=%d "
                "bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, size, bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Reduce(const void *sendbuf, void *recvbuf,
               int count, MPI_Datatype datatype,
               MPI_Op op, int root, MPI_Comm comm)
{
    size_t bytes = compute_bytes(count, datatype);
    int rank = -1, size = 0;
    if (is_mpi_initialized()) {
        PMPI_Comm_rank(comm, &rank);
        PMPI_Comm_size(comm, &size);
    }
    double t = is_mpi_initialized() ? PMPI_Wtime() : 0.0;

    if (logf) {
        fprintf(logf,
                "[%f] MPI_Reduce: world_rank=%d comm_rank=%d comm_size=%d "
                "root=%d bytes=%zu comm=%p\n",
                t, get_world_rank(), rank, size, root, bytes, (void*)comm);
        fflush(logf);
    }

    return PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}
