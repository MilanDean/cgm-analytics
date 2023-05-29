import Image from 'next/image'
import TopNav from './components/TopNav'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">

      <TopNav />

      <div className="relative flex place-items-center before:absolute before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px]">
        <Image
          className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
          src="/berkeley-bear.svg"
          alt="Next.js Logo"
          width={350}
          height={37}
          priority
        />
      </div>

      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify- dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
          <a
            className="pointer-events-none flex place-items-center gap-3 p-8 lg:pointer-events-auto lg:p-0"
            href="https://www.ischool.berkeley.edu/"
            target="/"
            rel="noopener noreferrer"
          >
            By:{' '}
            <Image
              src="/berkeleyischool-logo.svg"
              alt="UC Berkeley"
              className="dark:invert"
              width={150}
              height={18}
              priority
            />
          </a>
        </div>
      </div>

    </main>
  )
}
