#include "nicslu_util.h"
#include "nicslu.h"

#define ROW_LENGTH 1024

#define FAIL(code)	((code) < NICS_OK)

static int ReadHeader3(FILE *f, uint__t *m, uint__t *n, uint__t *nnz)
{
	char line[ROW_LENGTH];
	int read;
	line[0] = 0;
	*m = *n = *nnz = 0;

	do 
	{
		if (fgets(line, ROW_LENGTH-1, f) == NULL) return NICSLU_GENERAL_FAIL;
	} while (line[0] == '%');

#ifdef INT64__
#ifdef _WIN32
	if (sscanf(line, "%I64u %I64u %I64u", m, n, nnz) == 3)
#else
	if (sscanf(line, "%llu %llu %llu", m, n, nnz) == 3)
#endif
#else
	if (sscanf(line, "%u %u %u", m, n, nnz) == 3)
#endif
	{
		return NICS_OK;
	}
	else
	{
		do
		{ 
#ifdef INT64__
#ifdef _WIN32
			read = fscanf(f, "%I64u %I64u %I64u", m, n, nnz);
#else
			read = fscanf(f, "%llu %llu %llu", m, n, nnz);
#endif
#else
			read = fscanf(f, "%u %u %u", m, n, nnz);
#endif
			if (read == EOF) return NICSLU_GENERAL_FAIL;
		} while (read != 3);
	}

	return NICS_OK;
}

static int ReadHeader2(FILE *f, uint__t *m, uint__t *n, uint__t *nnz)
{
	char line[ROW_LENGTH];
	int read;
	line[0] = 0;
	*m = *n = *nnz = 0;

	do 
	{
		if (fgets(line, ROW_LENGTH-1, f) == NULL) return NICSLU_GENERAL_FAIL;
	} while (line[0] == '%');

#ifdef INT64__
#ifdef _WIN32
	if (sscanf(line, "%I64u %I64u", m, nnz) == 2)
#else
	if (sscanf(line, "%llu %llu", m, nnz) == 2)
#endif
#else
	if (sscanf(line, "%u %u", m, nnz) == 2)
#endif
	{
		*n = *m;
		return NICS_OK;
	}
	else
	{
		do
		{ 
#ifdef INT64__
#ifdef _WIN32
			read = fscanf(f, "%I64u %I64u", m, nnz);
#else
			read = fscanf(f, "%llu %llu", m, nnz);
#endif
#else
			read = fscanf(f, "%u %u", m, nnz);
#endif
			if (read == EOF) return NICSLU_GENERAL_FAIL;
		} while (read != 2);
	}
	*n = *m;

	return NICS_OK;
}

int NicsLU_ReadTripletColumnToSparse(char *file, uint__t *n, uint__t *nnz, \
									 real__t **ax, uint__t **ai, uint__t **ap)
{
	FILE *fp;
	int err;
	uint__t m, *aj, i, j;
	uint__t pre, cur, num;
	int cnt;

	if (NULL == file || NULL == n || NULL == nnz \
		|| NULL == ax || NULL == ai || NULL == ap) return NICSLU_ARGUMENT_ERROR;
	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}
	*n = *nnz = 0;

	fp = fopen(file, "r");
	if (NULL == fp) return NICSLU_FILE_CANNOT_OPEN;

	err = ReadHeader3(fp, n, &m, nnz);
	if (FAIL(err))
	{
		fclose(fp);
		return err;
	}
	if (m != *n)
	{
		fclose(fp);
		return NICSLU_MATRIX_INVALID;
	}

	aj = (uint__t *)malloc(sizeof(uint__t)*(*nnz));
	*ax = (real__t *)malloc(sizeof(real__t)*(*nnz));
	*ai = (uint__t *)malloc(sizeof(uint__t)*(*nnz));
	*ap = (uint__t *)malloc(sizeof(uint__t)*(m+1));
	if (NULL == aj || NULL == *ax || NULL == *ai || NULL == *ap)
	{
		fclose(fp);
		if (aj != NULL) free(aj);
		if (*ax != NULL)
		{
			*ax = NULL;
			free(*ax);
		}
		if (*ai != NULL)
		{
			*ai = NULL;
			free(*ai);
		}
		if (*ap != NULL)
		{
			*ap = NULL;
			free(*ap);
		}
		return NICSLU_MEMORY_OVERFLOW;
	}

	for (i=0; i<*nnz; ++i)
	{
#ifdef INT64__
#ifdef _WIN32
		cnt = fscanf(fp, "%I64u %I64u %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#else
		cnt = fscanf(fp, "%llu %llu %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#endif
#else
		cnt = fscanf(fp, "%u %u %lf", &((*ai)[i]), &aj[i], &((*ax)[i]));
#endif
		if (cnt != 3)
		{
			free(aj);
			fclose(fp);
			return NICSLU_MATRIX_INVALID;
		}

		--((*ai)[i]);
		--aj[i];

		if ((*ai)[i] >= *n || aj[i] >= *n)
		{
			free(aj);
			fclose(fp);
			return NICSLU_MATRIX_INVALID;
		}
	}
	fclose(fp);

	(*ap)[0] = 0;
	pre = 0;
	cur = 0;
	num = 0;

	for (i=0; i<*nnz; ++i)
	{
		cur = aj[i];
		if (pre == cur)
		{
			++num;
		}
		else
		{
			num += (*ap)[pre];
			for (j=pre+1; j<=cur; j++)
			{
				(*ap)[j] = num;
			}
			pre = cur;
			num = 1;
		}
	}
	num += (*ap)[cur];
	for (i=cur+1; i<=m; i++)
	{
		(*ap)[i] = num;
	}

	free(aj);
	return NICS_OK;
}

int NicsLU_ReadTripletRowToSparse(char *file, uint__t *n, uint__t *nnz, \
									 real__t **ax, uint__t **ai, uint__t **ap)
{
	FILE *fp;
	int err;
	uint__t m, *aj, i, j;
	uint__t pre, cur, num;
	int cnt;

	if (NULL == file || NULL == n || NULL == nnz \
		|| NULL == ax || NULL == ai || NULL == ap) return NICSLU_ARGUMENT_ERROR;
	if (*ax != NULL)
	{
		free(*ax);
		*ax = NULL;
	}
	if (*ai != NULL)
	{
		free(*ai);
		*ai = NULL;
	}
	if (*ap != NULL)
	{
		free(*ap);
		*ap = NULL;
	}
	*n = *nnz = 0;

	fp = fopen(file, "r");
	if (NULL == fp) return NICSLU_FILE_CANNOT_OPEN;

	err = ReadHeader3(fp, n, &m, nnz);
	if (FAIL(err))
	{
		fclose(fp);
		return err;
	}
	if (m != *n)
	{
		fclose(fp);
		return NICSLU_MATRIX_INVALID;
	}

	aj = (uint__t *)malloc(sizeof(uint__t)*(*nnz));
	*ax = (real__t *)malloc(sizeof(real__t)*(*nnz));
	*ai = (uint__t *)malloc(sizeof(uint__t)*(*nnz));
	*ap = (uint__t *)malloc(sizeof(uint__t)*(m+1));
	if (NULL == aj || NULL == *ax || NULL == *ai || NULL == *ap)
	{
		fclose(fp);
		if (aj != NULL) free(aj);
		if (*ax != NULL)
		{
			*ax = NULL;
			free(*ax);
		}
		if (*ai != NULL)
		{
			*ai = NULL;
			free(*ai);
		}
		if (*ap != NULL)
		{
			*ap = NULL;
			free(*ap);
		}
		return NICSLU_MEMORY_OVERFLOW;
	}

	for (i=0; i<*nnz; ++i)
	{
#ifdef INT64__
#ifdef _WIN32
		cnt = fscanf(fp, "%I64u %I64u %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#else
		cnt = fscanf(fp, "%llu %llu %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#endif
#else
		cnt = fscanf(fp, "%u %u %lf", &aj[i], &((*ai)[i]), &((*ax)[i]));
#endif
		if (cnt != 3)
		{
			free(aj);
			fclose(fp);
			return NICSLU_MATRIX_INVALID;
		}

		--((*ai)[i]);
		--aj[i];

		if ((*ai)[i] >= *n || aj[i] >= *n)
		{
			free(aj);
			fclose(fp);
			return NICSLU_MATRIX_INVALID;
		}
	}
	fclose(fp);

	(*ap)[0] = 0;
	pre = 0;
	cur = 0;
	num = 0;

	for (i=0; i<*nnz; ++i)
	{
		cur = aj[i];
		if (pre == cur)
		{
			++num;
		}
		else
		{
			num += (*ap)[pre];
			for (j=pre+1; j<=cur; j++)
			{
				(*ap)[j] = num;
			}
			pre = cur;
			num = 1;
		}
	}
	num += (*ap)[cur];
	for (i=cur+1; i<=m; i++)
	{
		(*ap)[i] = num;
	}

	free(aj);
	return NICS_OK;
}
