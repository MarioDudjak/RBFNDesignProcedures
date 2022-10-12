using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Nelder_Mead
{
    class Program
    {
        static Random RndGen = new Random();

        static void Main(string[] args)
        {
            NMalgorithm nmalg = new NMalgorithm(10, 10000, 0.5, -0.5, 0.5, 1, 2, -100, 100, RndGen);

            Console.WriteLine(nmalg.Run());
        }
    }
}
