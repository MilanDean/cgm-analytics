const people = [
    {
      name: "Dimitrios Psaltos",
      role: "Data Scientist",
      imageUrl: "https://media.licdn.com/dms/image/C5603AQGdSqokI-fiTw/profile-displayphoto-shrink_400_400/0/1517397971915?e=1694649600&v=beta&t=8ASXS2qApzFETThBfrpVxAezq60t5zl7kySfslMAUiQ",
      info: "Dimitrios is a biomedical engineer with knowledge and experience in the pharmaceutical industry. Dimitrios serves as an Senior Associate Data Scientist in Pfizer's Digital Medicine and Translational Imaging group (DMTI) which helps implement digital biomarkers and wearable sensing technologies into clinical studies across multiple therapeutic areas."
    },
    {
      name: 'Marguerite Morgan',
      role: 'Data Scientist',
      imageUrl: 'https://media.licdn.com/dms/image/C4D03AQEZ_wpzOaWYFw/profile-displayphoto-shrink_400_400/0/1629753708632?e=1694649600&v=beta&t=kXiokUTywFbgK26IvQSiwp2-PDHkD1kOT9OkovRzY-8',
      info: ''
    },
    {
      name: 'Arun Surendranath',
      role: 'Data Scientist',
      imageUrl: 'https://media.licdn.com/dms/image/C4D03AQH7Ru-KNVEPpw/profile-displayphoto-shrink_400_400/0/1516570547338?e=1694649600&v=beta&t=Y8qzPPPcMacw9pW8AXQ6X0eSXkd9on4CNlgPD90hX5g',
      info: 'Arun is a Materials scientist at the intersection of materials development and data science for energy storage solutions. experienced in bringing new material development into market with innovative process development methodologies. well versed in leading cross functional teams to bring out the best to solve problems.'
    },
    {
      name: 'Milan Dean',
      role: 'Software Engineer - ML',
      imageUrl: 'https://media.licdn.com/dms/image/C4E03AQFdWfNAO7l2LQ/profile-displayphoto-shrink_400_400/0/1635734084511?e=1694649600&v=beta&t=v4lnLtCxkwUv3LDm0vfMVitA2nNvoP4PJJHQzBG4AfM',
      info: 'Milan is a Software Development Engineer at Intel focused on developing a cloud-based SaaS specializing in system performance profiling. Prior to this role, Milan served as Lead Data Scientist on the Validation Capacity & Controls team within the Internal Validation Engineering (iVE) Division at Intel, building internal ML tools focused on predictive forecasting.'
    },
  ]
  
  export default function TeamPage() {
    return (
      <div className="bg-white py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl sm:text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Meet the Team</h2>
            <p className="mt-6 text-lg leading-8 text-gray-600">
              We’re a dynamic group of individuals who are passionate about what we do and what we build!
            </p>
          </div>
          <ul
            role="list"
            className="mx-auto mt-20 grid max-w-2xl grid-cols-1 gap-x-6 gap-y-20 sm:grid-cols-2 lg:max-w-4xl lg:gap-x-8 xl:max-w-none"
          >
            {people.map((person) => (
              <li key={person.name} className="flex flex-col gap-6 xl:flex-row">
                <img className="aspect-[4/5] w-52 flex-none rounded-2xl object-cover" src={person.imageUrl} alt="" />
                <div className="flex-auto">
                  <h3 className="text-lg font-semibold leading-8 tracking-tight text-gray-900">{person.name}</h3>
                  <p className="text-base leading-7 text-gray-600">{person.role}</p>
                  <p className="mt-6 text-base leading-7 text-gray-600">{person.info}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    )
  }
