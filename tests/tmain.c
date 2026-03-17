/* tmain.c -- Moa test runner
 * Runs all registered tests. Like a headcount after a field trip,
 * except the children are unit tests and some of them may be dead. */

#include "tharns.h"

tcase_t th_list[TH_MAXTS];
int th_cnt = 0;
int npass  = 0;
int nfail  = 0;
int nskip  = 0;

int
main(int argc, char **argv)
{
    int i;
    const char *filter = NULL;

    if (argc > 1) filter = argv[1];

    printf("moa tests: %d registered\n\n", th_cnt);

    for (i = 0; i < th_cnt; i++) {
        /* If filter given, match against category or name */
        if (filter) {
            if (strstr(th_list[i].tcats, filter) == NULL &&
                strstr(th_list[i].tname, filter) == NULL)
                continue;
        }

        int pre_fail = nfail;
        printf("  [%s] %s ... ", th_list[i].tcats, th_list[i].tname);
        fflush(stdout);
        th_list[i].func();
        if (nfail == pre_fail) {
            printf("ok\n");
            npass++;
        }
    }

    printf("\n  %d pass, %d fail, %d skip\n",
           npass, nfail, nskip);

    return nfail > 0 ? 1 : 0;
}
