import Link from 'next/link';

export default function Example() {
  return (
    <div className="relative bg-white">
      <div className="mx-auto max-w-7xl lg:grid lg:grid-cols-12 lg:gap-x-8 lg:px-8">
        <div className="px-6 pb-24 pt-10 sm:pb-32 lg:col-span-7 lg:px-0 lg:pb-56 lg:pt-48 xl:col-span-6">
          <div className="mx-auto sm:mt-10 max-w-2xl">
            <img
              className="w-1/2"
              src='./NavLogo.png'
              alt="Nutrinet"
            />
          </div>
          <div className="mx-auto max-w-2xl lg:mx-0">
            <h1 className="font-bold tracking-tight text-gray-600 sm:mt-10 sm:text-6xl">
              ML-powered insights for your health journey.
            </h1>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              NutriNet leverages Machine Learning to transform personal glucose data into actionable insights, 
              empowering individuals to manage their Type 1 Diabetes.
            </p>
            <div className="mt-10 flex items-center gap-x-6">
            <Link href="/upload">
              <span
                className="rounded-md bg-green-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-green-600"
              >
                Get started
              </span>
            </Link>
            <Link href="/team">
              <span className="text-sm font-semibold leading-6 text-gray-900">
                Meet the Team 
                <span aria-hidden="true"> → </span>
              </span>
            </Link>
            </div>
          </div>
        </div>
        <div className="relative lg:col-span-5 lg:-mr-8 xl:absolute xl:inset-0 xl:left-1/2 xl:mr-0">
          <img
            className="aspect-[3/2] w-full bg-gray-50 object-cover lg:absolute lg:inset-0 lg:aspect-auto lg:h-full"
            src="https://images.unsplash.com/photo-1498758536662-35b82cd15e29?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2102&q=80"
            alt=""
          />
        </div>
      </div>
    </div>
  )
}

